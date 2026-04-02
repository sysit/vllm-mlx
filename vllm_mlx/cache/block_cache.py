# SPDX-License-Identifier: Apache-2.0
"""
Block Cache for vllm-mlx.

Block-based prefix cache that uses PagedCache for block-level storage.
Provides efficient prefix sharing with Copy-on-Write (COW) support.
"""

import copy
import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

from .paged_cache import BlockTable, PagedCache

logger = logging.getLogger(__name__)


@dataclass
class BlockCacheEntry:
    """Entry mapping a token sequence to cache blocks."""

    block_table: BlockTable
    cache_data: List[Any]  # Actual KV cache data per block
    last_access: float


class BlockCache:
    """
    Block-based prefix cache using PagedCache for storage.

    Features:
    - Block-level prefix sharing (64 tokens per block)
    - Copy-on-Write for efficient forking
    - Hash-based deduplication across requests
    - Reference counting for memory efficiency

    This is the recommended cache for production use when memory
    efficiency for concurrent requests is important.

    Example:
        paged_cache = PagedCache(block_size=64, max_blocks=1000)
        cache = BlockCache(model, paged_cache)

        # Check for cached prefix
        block_table, remaining_tokens = cache.fetch(request_id, tokens)

        # After generation, store cache
        cache.store(request_id, tokens, kv_cache_data)

        # Clean up when request completes
        cache.release(request_id)
    """

    def __init__(
        self,
        model: Any,
        paged_cache: PagedCache = None,
        paged_cache_manager: PagedCache = None,  # Backward compatibility
    ):
        """
        Initialize block cache.

        Args:
            model: The MLX model (used for identification)
            paged_cache: The PagedCache instance for block management
            paged_cache_manager: Legacy alias for paged_cache (backward compatibility)
        """
        # Handle backward compatibility: paged_cache_manager -> paged_cache
        if paged_cache is None and paged_cache_manager is not None:
            paged_cache = paged_cache_manager
        if paged_cache is None:
            raise ValueError("Either paged_cache or paged_cache_manager must be provided")
            
        self.model = model
        self.model_key = id(model)
        self.paged_cache = paged_cache
        self.block_size = paged_cache.block_size

        # Hash table for quick prefix lookup
        # Maps hash(tokens[:block_size*n]) -> (tokens, block_ids)
        self._prefix_index: Dict[str, Tuple[List[int], List[int]]] = {}

        # Request to block table mapping
        self._request_tables: Dict[str, BlockCacheEntry] = {}

        # Statistics
        self._hits = 0
        self._misses = 0
        self._tokens_saved = 0

    def fetch(
        self,
        request_id: str,
        tokens: List[int],
    ) -> Tuple[Optional[BlockTable], List[int]]:
        """
        Find cached prefix blocks for the given tokens.

        Args:
            request_id: Unique request identifier
            tokens: Input token sequence

        Returns:
            Tuple of (block_table, remaining_tokens)
            - block_table: BlockTable if prefix found, None otherwise
            - remaining_tokens: Tokens that need processing
        """
        if not tokens:
            return None, tokens

        # Try to find shared prefix blocks
        shared_block_ids, remaining = self.paged_cache.find_shared_prefix(tokens)

        if shared_block_ids:
            # Create block table for this request with shared blocks
            block_table = self.paged_cache.create_block_table(request_id)

            for block_id in shared_block_ids:
                # Increment ref count for sharing
                self.paged_cache.increment_ref(block_id)
                block = self.paged_cache.allocated_blocks.get(block_id)
                if block:
                    block_table.block_ids.append(block_id)
                    block_table.num_tokens += block.token_count

            num_prefix_tokens = len(tokens) - len(remaining)
            self._hits += 1
            self._tokens_saved += num_prefix_tokens

            logger.debug(
                f"Cache hit for {request_id}: "
                f"{len(shared_block_ids)} blocks, {num_prefix_tokens} tokens"
            )

            return block_table, remaining

        # Try prefix index for longer matches
        best_match = self._find_best_prefix_match(tokens)
        if best_match:
            matched_tokens, matched_block_ids = best_match

            # Fork the matched blocks
            block_table = self.paged_cache.create_block_table(request_id)
            for block_id in matched_block_ids:
                self.paged_cache.increment_ref(block_id)
                block = self.paged_cache.allocated_blocks.get(block_id)
                if block:
                    block_table.block_ids.append(block_id)
                    block_table.num_tokens += block.token_count

            remaining = tokens[len(matched_tokens) :]
            self._hits += 1
            self._tokens_saved += len(matched_tokens)

            logger.debug(
                f"Prefix index hit for {request_id}: "
                f"{len(matched_tokens)} tokens matched"
            )

            return block_table, remaining

        # No cache hit
        self._misses += 1
        logger.debug(f"Cache miss for {request_id}")
        return None, tokens

    def store(
        self,
        request_id: str,
        tokens: List[int],
        cache_data: List[Any],
    ) -> Optional[BlockTable]:
        """
        Store computed cache for future reuse.

        This method stores actual tensor data (not references) when cache_data
        contains extracted states from mlx-lm's KVCache.state property.

        Args:
            request_id: Unique request identifier
            tokens: Token sequence that was processed
            cache_data: The computed KV cache to store. Can be:
                - List of KVCache objects (legacy, stores references)
                - List of dicts with 'state': (keys, values) tensors (new, stores slices)

        Returns:
            BlockTable for the stored cache, or None on failure
        """
        if not tokens:
            return None

        # Check if cache_data contains extracted tensor states
        is_tensor_data = (
            cache_data
            and isinstance(cache_data, list)
            and len(cache_data) > 0
            and isinstance(cache_data[0], dict)
            and "state" in cache_data[0]
        )

        # Get or create block table
        block_table = self.paged_cache.get_block_table(request_id)
        if not block_table:
            block_table = self.paged_cache.create_block_table(request_id)

        # Determine tokens we need to cache (not already in block_table)
        existing_tokens = block_table.num_tokens
        new_tokens = tokens[existing_tokens:]

        if not new_tokens:
            # All tokens already cached
            return block_table

        # Allocate blocks for new tokens
        num_new_blocks = (len(new_tokens) + self.block_size - 1) // self.block_size

        for i in range(num_new_blocks):
            start_idx = i * self.block_size
            end_idx = min(start_idx + self.block_size, len(new_tokens))
            block_tokens = new_tokens[start_idx:end_idx]

            # Token range in the original sequence (accounting for existing tokens)
            global_start = existing_tokens + start_idx
            global_end = existing_tokens + end_idx

            # Check if this block already exists (deduplication)
            if len(block_tokens) == self.block_size:
                existing_block = self.paged_cache.find_cached_block(block_tokens)
                if existing_block:
                    # Reuse existing block
                    self.paged_cache.increment_ref(existing_block.block_id)
                    block_table.block_ids.append(existing_block.block_id)
                    block_table.num_tokens += len(block_tokens)
                    continue

            # Allocate new block
            block = self.paged_cache.allocate_block()
            if not block:
                # Handle memory pressure
                if not self.paged_cache.handle_memory_pressure(1):
                    logger.warning(f"Cannot allocate block for {request_id}")
                    break
                block = self.paged_cache.allocate_block()
                if not block:
                    break

            # Store block data
            block.token_count = len(block_tokens)
            block_table.block_ids.append(block.block_id)
            block_table.num_tokens += len(block_tokens)

            # Extract and store actual tensor slices for this block
            if is_tensor_data and HAS_MLX:
                block_kv_data = self._extract_block_tensor_slice(
                    cache_data, global_start, global_end
                )
                if block_kv_data:
                    block.cache_data = block_kv_data
                    logger.debug(
                        f"Stored tensor slice for block {block.block_id}: "
                        f"tokens [{global_start}:{global_end}], {len(block_kv_data)} layers"
                    )

            # Register hash for full blocks (for deduplication)
            if len(block_tokens) == self.block_size:
                self.paged_cache.register_block_hash(block, block_tokens)

        # Update prefix index
        self._update_prefix_index(tokens, block_table.block_ids)

        # Store entry for request (for legacy compatibility)
        self._request_tables[request_id] = BlockCacheEntry(
            block_table=block_table,
            cache_data=cache_data,
            last_access=time.time(),
        )

        blocks_with_data = sum(
            1
            for bid in block_table.block_ids
            if self.paged_cache.allocated_blocks.get(bid)
            and self.paged_cache.allocated_blocks[bid].cache_data is not None
        )

        logger.debug(
            f"Stored cache for {request_id}: "
            f"{len(block_table.block_ids)} blocks ({blocks_with_data} with tensor data), "
            f"{block_table.num_tokens} tokens"
        )

        return block_table

    def _extract_block_tensor_slice(
        self,
        cache_data: List[Dict[str, Any]],
        start_idx: int,
        end_idx: int,
    ) -> Optional[List[Tuple[Any, Any]]]:
        """
        Extract tensor slices for a single block from cache data.

        Args:
            cache_data: List of layer states, each containing 'state': (keys, values)
            start_idx: Start token index in the sequence
            end_idx: End token index in the sequence

        Returns:
            List of (keys_slice, values_slice) for each layer, or None on failure
        """
        if not HAS_MLX or not cache_data:
            return None

        try:
            block_slices = []
            for layer_state in cache_data:
                if "state" not in layer_state:
                    continue

                keys, values = layer_state["state"]

                # KV cache shape: (batch, n_kv_heads, seq_len, head_dim)
                # Slice along seq_len dimension (axis 2)
                seq_len = keys.shape[2] if hasattr(keys, "shape") else 0

                if end_idx > seq_len:
                    # Requested range extends beyond available data
                    logger.debug(
                        f"Block slice [{start_idx}:{end_idx}] exceeds seq_len {seq_len}"
                    )
                    # Use whatever is available
                    actual_end = min(end_idx, seq_len)
                    if start_idx >= actual_end:
                        continue
                    keys_slice = keys[:, :, start_idx:actual_end, :]
                    values_slice = values[:, :, start_idx:actual_end, :]
                else:
                    keys_slice = keys[:, :, start_idx:end_idx, :]
                    values_slice = values[:, :, start_idx:end_idx, :]

                block_slices.append((keys_slice, values_slice))

            return block_slices if block_slices else None

        except Exception as e:
            logger.warning(f"Failed to extract block tensor slice: {e}")
            return None

    def get_cache_for_generation(
        self,
        request_id: str,
    ) -> Tuple[Optional[List[Any]], bool]:
        """
        Get cache data for generation, applying COW if needed.

        Args:
            request_id: Request identifier

        Returns:
            Tuple of (cache_data, was_copied)
        """
        entry = self._request_tables.get(request_id)
        if not entry:
            return None, False

        # Get blocks with COW
        blocks, was_copied = self.paged_cache.get_blocks_for_generation(
            entry.block_table
        )

        if was_copied:
            # Deep copy cache data for modified blocks
            cache_data = copy.deepcopy(entry.cache_data)
        else:
            cache_data = entry.cache_data

        entry.last_access = time.time()
        return cache_data, was_copied

    def release(self, request_id: str) -> None:
        """
        Release cache blocks for a completed request.

        Args:
            request_id: Request identifier
        """
        entry = self._request_tables.pop(request_id, None)
        if entry:
            self.paged_cache.delete_block_table(request_id)
            logger.debug(f"Released cache for {request_id}")

    def fork(
        self,
        source_request_id: str,
        new_request_id: str,
    ) -> Optional[BlockTable]:
        """
        Fork cache from one request to another (COW).

        Args:
            source_request_id: Source request ID
            new_request_id: New request ID

        Returns:
            Forked BlockTable, or None if source not found
        """
        source_entry = self._request_tables.get(source_request_id)
        if not source_entry:
            return None

        # Fork block table (increments ref counts)
        forked_table = self.paged_cache.fork_block_table(
            source_entry.block_table,
            new_request_id,
        )

        # Create new entry with reference to same cache data
        self._request_tables[new_request_id] = BlockCacheEntry(
            block_table=forked_table,
            cache_data=source_entry.cache_data,  # Shared reference
            last_access=time.time(),
        )

        logger.debug(f"Forked cache: {source_request_id} -> {new_request_id}")

        return forked_table

    def reconstruct(
        self,
        block_table: BlockTable,
    ) -> Optional[List[Any]]:
        """
        Reconstruct KVCache objects from stored block tensor data.

        This method concatenates tensor slices from all blocks and
        creates new KVCache objects that can be used for inference.

        Args:
            block_table: BlockTable containing block IDs to reconstruct from

        Returns:
            List of reconstructed KVCache objects (one per layer),
            or None if reconstruction fails
        """
        if not block_table or not block_table.block_ids:
            return None

        if not HAS_MLX:
            logger.warning("Cannot reconstruct cache: MLX not available")
            return None

        try:
            # Collect cache data from all blocks
            all_block_data = []
            for block_id in block_table.block_ids:
                block = self.paged_cache.allocated_blocks.get(block_id)
                if not block:
                    logger.warning(f"Block {block_id} not found in allocated blocks")
                    return None

                if block.cache_data is None:
                    logger.debug(f"Block {block_id} has no tensor data stored")
                    return None

                all_block_data.append(block.cache_data)

            if not all_block_data:
                return None

            # Get number of layers from first block
            num_layers = len(all_block_data[0])
            if num_layers == 0:
                return None

            # Concatenate tensors for each layer
            reconstructed_caches = []

            for layer_idx in range(num_layers):
                layer_keys = []
                layer_values = []

                for block_data in all_block_data:
                    if layer_idx < len(block_data):
                        keys_slice, values_slice = block_data[layer_idx]
                        layer_keys.append(keys_slice)
                        layer_values.append(values_slice)

                if not layer_keys:
                    continue

                # Concatenate along sequence dimension (axis 2)
                # Shape: (batch, n_kv_heads, seq_len, head_dim)
                concat_keys = mx.concatenate(layer_keys, axis=2)
                concat_values = mx.concatenate(layer_values, axis=2)

                # Create KVCache object
                # Try to use mlx_lm's KVCache.from_state if available
                try:
                    from mlx_lm.models.cache import KVCache

                    # Create new cache and set its state
                    cache = KVCache()
                    seq_len = concat_keys.shape[2]

                    # Set internal state directly
                    # KVCache stores keys/values and offset
                    cache.keys = concat_keys
                    cache.values = concat_values
                    cache.offset = seq_len

                    reconstructed_caches.append(cache)

                except ImportError:
                    # Fallback: create a simple cache-like object
                    class SimpleKVCache:
                        def __init__(self, keys, values):
                            self.keys = keys
                            self.values = values
                            self.offset = keys.shape[2]

                        @property
                        def state(self):
                            return (self.keys, self.values)

                        @property
                        def meta_state(self):
                            return (str(self.offset),)

                    cache = SimpleKVCache(concat_keys, concat_values)
                    reconstructed_caches.append(cache)

            if not reconstructed_caches:
                return None

            logger.debug(
                f"Reconstructed cache: {len(reconstructed_caches)} layers, "
                f"{block_table.num_tokens} tokens from {len(block_table.block_ids)} blocks"
            )

            return reconstructed_caches

        except Exception as e:
            logger.warning(f"Failed to reconstruct cache: {e}")
            import traceback

            logger.debug(traceback.format_exc())
            return None

    def _find_best_prefix_match(
        self,
        tokens: List[int],
    ) -> Optional[Tuple[List[int], List[int]]]:
        """Find best matching prefix in the index."""
        best_match = None
        best_len = 0

        # Try progressively longer prefixes
        for num_blocks in range(1, len(tokens) // self.block_size + 1):
            prefix_len = num_blocks * self.block_size
            if prefix_len > len(tokens):
                break

            prefix_tokens = tokens[:prefix_len]
            prefix_hash = self.paged_cache.compute_block_hash(prefix_tokens)

            if prefix_hash in self._prefix_index:
                cached_tokens, block_ids = self._prefix_index[prefix_hash]
                if cached_tokens == prefix_tokens and len(cached_tokens) > best_len:
                    best_match = (cached_tokens, block_ids)
                    best_len = len(cached_tokens)

        return best_match

    def _update_prefix_index(
        self,
        tokens: List[int],
        block_ids: List[int],
    ) -> None:
        """Update prefix index with new token sequence."""
        # Index block-aligned prefixes
        for i in range(1, len(block_ids) + 1):
            prefix_len = min(i * self.block_size, len(tokens))
            prefix_tokens = tokens[:prefix_len]
            prefix_hash = self.paged_cache.compute_block_hash(prefix_tokens)
            self._prefix_index[prefix_hash] = (prefix_tokens, block_ids[:i])

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        paged_stats = self.paged_cache.get_memory_usage()
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": (
                self._hits / (self._hits + self._misses)
                if (self._hits + self._misses) > 0
                else 0
            ),
            "tokens_saved": self._tokens_saved,
            "active_requests": len(self._request_tables),
            **paged_stats,
        }

    def reset_stats(self) -> None:
        """Reset statistics."""
        self._hits = 0
        self._misses = 0
        self._tokens_saved = 0
        self.paged_cache.reset_stats()

    def clear(self) -> None:
        """Clear all cached data."""
        self._request_tables.clear()
        self._prefix_index.clear()
        self.paged_cache.clear()
        self.reset_stats()

    def __len__(self) -> int:
        """Return number of active request entries."""
        return len(self._request_tables)

    # Backward compatibility method aliases
    def fetch_cache(
        self,
        request_id: str,
        tokens: List[int],
    ) -> Tuple[Optional[BlockTable], List[int]]:
        """Alias for fetch() for backward compatibility."""
        return self.fetch(request_id, tokens)

    def store_cache(
        self,
        request_id: str,
        tokens: List[int],
        cache_data: List[Any],
    ) -> Optional[BlockTable]:
        """Alias for store() for backward compatibility."""
        return self.store(request_id, tokens, cache_data)

    def release_cache(self, request_id: str) -> None:
        """Alias for release() for backward compatibility."""
        self.release(request_id)

    def fork_cache(
        self,
        source_request_id: str,
        new_request_id: str,
    ) -> Optional[BlockTable]:
        """Alias for fork() for backward compatibility."""
        return self.fork(source_request_id, new_request_id)

    def reconstruct_cache(
        self,
        block_table: BlockTable,
    ) -> Optional[List[Any]]:
        """Alias for reconstruct() for backward compatibility."""
        return self.reconstruct(block_table)


# =============================================================================
# Backward Compatibility Alias
# =============================================================================

BlockAwarePrefixCache = BlockCache