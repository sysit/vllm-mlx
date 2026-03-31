# SPDX-License-Identifier: Apache-2.0
"""
KVCacheBackend - Standard LLM text cache backend.

Provides KV cache management for standard transformer models:
- Batch support via KVCache.merge() and BatchKVCache.extract()
- Prefix hash-based lookup and reuse
- Memory-aware eviction policies
- Block-based cache with CoW prefix sharing (optional)
"""

import hashlib
import logging
import time
from typing import Any, Optional, List, Dict, Tuple, TYPE_CHECKING

from .backend import CacheBackend, CacheEntry

if TYPE_CHECKING:
    from .block_manager import BlockManager
    from .cow import CoWPrefixSharing, PrefixSharingIntegration

logger = logging.getLogger(__name__)


class KVCacheBackend(CacheBackend):
    """
    KV cache backend for standard LLM text models.

    Supports:
    - Batch operations (merge multiple single-sequence caches)
    - Prefix hash-based cache reuse
    - Memory-aware eviction
    - Compatible with mlx-lm's BatchKVCache
    - Optional Block-based cache with CoW prefix sharing

    This backend handles pure transformer models that use KVCache
    for attention layer caching. It supports merging individual
    request caches into a batched cache for efficient generation.

    With BlockManager Integration:
        The backend can use block-based allocation for more efficient
        prefix sharing across requests. Enable with use_block_manager=True.
    """

    def __init__(
        self,
        max_entries: int = 100,
        max_memory_mb: Optional[int] = None,
        eviction_policy: str = "lru",  # "lru", "lfu", "fifo"
        use_block_manager: bool = False,
        block_size: int = 16,
        max_blocks: int = 1000,
    ):
        """
        Initialize KV cache backend.

        Args:
            max_entries: Maximum number of cache entries
            max_memory_mb: Maximum memory budget in MB (None = unlimited)
            eviction_policy: Cache eviction policy
            use_block_manager: Enable block-based cache management
            block_size: Tokens per block (default: 16)
            max_blocks: Maximum blocks for block manager
        """
        self.max_entries = max_entries
        self.max_memory_mb = max_memory_mb
        self.eviction_policy = eviction_policy
        self.use_block_manager = use_block_manager

        self._entries: Dict[str, CacheEntry] = {}
        self._hits = 0
        self._misses = 0
        self._evictions = 0

        # Block-based cache management (optional)
        self._block_manager: Optional["BlockManager"] = None
        self._cow: Optional["CoWPrefixSharing"] = None
        self._prefix_integration: Optional["PrefixSharingIntegration"] = None

        if use_block_manager:
            from .block_manager import BlockManager
            from .cow import CoWPrefixSharing, PrefixSharingIntegration

            self._block_manager = BlockManager(
                max_blocks=max_blocks,
                block_size=block_size,
            )
            self._cow = CoWPrefixSharing(self._block_manager)
            self._prefix_integration = PrefixSharingIntegration(self._block_manager)

            logger.info(
                f"[KVCacheBackend] Block-based cache enabled: "
                f"block_size={block_size}, max_blocks={max_blocks}"
            )

    def _compute_hash(self, tokens: List[int]) -> str:
        """
        Compute prefix hash from token IDs.

        Args:
            tokens: List of token IDs

        Returns:
            Hash string for the prefix
        """
        token_bytes = b"".join(t.to_bytes(4, "little") for t in tokens)
        return hashlib.sha256(token_bytes).hexdigest()[:16]

    def lookup(self, prefix_hash: str) -> Optional[CacheEntry]:
        """
        Look up KV cache by prefix hash.

        Args:
            prefix_hash: Hash of prefix tokens

        Returns:
            CacheEntry if found and valid, None otherwise
        """
        entry = self._entries.get(prefix_hash)
        if entry is None:
            self._misses += 1
            return None

        # Validate entry before returning
        if not entry.is_valid():
            logger.debug(f"[KVCacheBackend] Entry {prefix_hash} invalid, removing")
            self.remove(prefix_hash)
            self._misses += 1
            return None

        # Update access stats
        entry.update_access(time.time())
        self._hits += 1
        return entry

    def store(self, prefix_hash: str, entry: CacheEntry) -> None:
        """
        Store KV cache entry.

        Args:
            prefix_hash: Hash of prefix tokens
            entry: Cache entry to store
        """
        # Validate entry
        if not entry.is_valid():
            logger.warning(f"[KVCacheBackend] Attempted to store invalid entry")
            return

        # Check capacity and evict if needed
        self._maybe_evict()

        # Store entry
        entry.created_at = time.time()
        entry.last_accessed = time.time()
        entry.cache_type = "kv"
        self._entries[prefix_hash] = entry

        logger.debug(
            f"[KVCacheBackend] Stored entry {prefix_hash}, "
            f"tokens={entry.tokens}, size={entry.size_bytes / 1024:.1f}KB"
        )

    def supports_batch(self) -> bool:
        """
        KV cache supports batch operations.

        Returns:
            True - KV cache can merge/extract
        """
        return True

    def merge(self, entries: List[CacheEntry]) -> CacheEntry:
        """
        Merge multiple single-sequence KV caches into a batched cache.

        This uses mlx-lm's KVCache.merge() to create a BatchKVCache
        with proper left-padding alignment.

        Args:
            entries: List of single-sequence CacheEntry objects

        Returns:
            New batched CacheEntry

        Raises:
            ValueError: If entries are incompatible
        """
        if not entries:
            raise ValueError("Cannot merge empty entry list")

        if not self.supports_batch():
            raise NotImplementedError("KVCacheBackend must support batch")

        # Validate all entries are single-sequence
        for entry in entries:
            for block in entry.blocks:
                if hasattr(block, "keys") and block.keys is not None:
                    if block.keys.shape[0] != 1:
                        raise ValueError(
                            f"Cannot merge multi-sequence cache: "
                            f"keys.shape[0]={block.keys.shape[0]}"
                        )

        # Use first entry's structure, merge each layer
        try:
            from mlx_lm.models.cache import KVCache, BatchKVCache

            merged_blocks = []
            for layer_idx in range(len(entries[0].blocks)):
                layer_caches = [e.blocks[layer_idx] for e in entries]
                # KVCache.merge() returns BatchKVCache
                merged_block = layer_caches[0].merge(layer_caches)
                merged_blocks.append(merged_block)

            # Create merged entry
            merged_entry = CacheEntry(
                entry_id=f"merged_{len(entries)}_{time.time():.0f}",
                prefix_hash="merged",  # Batched entries don't have single hash
                cache_type="kv",
                blocks=merged_blocks,
                tokens=sum(e.tokens for e in entries),
            )

            logger.debug(
                f"[KVCacheBackend] Merged {len(entries)} entries, "
                f"total_tokens={merged_entry.tokens}"
            )
            return merged_entry

        except Exception as e:
            logger.error(f"[KVCacheBackend] Merge failed: {e}")
            raise ValueError(
                f"Failed to merge KV caches: {type(e).__name__}: {e}. "
                f"Check that all entries have compatible structure."
            )

    def extract(self, entry: CacheEntry, idx: int) -> CacheEntry:
        """
        Extract single-sequence cache from batched entry.

        Args:
            entry: Batched cache entry
            idx: Index to extract (0-based)

        Returns:
            New single-sequence CacheEntry
        """
        if not self.supports_batch():
            raise NotImplementedError("KVCacheBackend must support batch")

        extracted_blocks = []
        for block in entry.blocks:
            if hasattr(block, "extract"):
                extracted_blocks.append(block.extract(idx))
            else:
                # Fallback: slice if possible
                if hasattr(block, "keys") and block.keys is not None:
                    # This shouldn't happen for proper BatchKVCache
                    logger.warning(
                        "[KVCacheBackend] Block has no extract method, "
                        "using slice fallback"
                    )
                    extracted_blocks.append(block)
                else:
                    extracted_blocks.append(block)

        return CacheEntry(
            entry_id=f"extracted_{idx}_{entry.entry_id}",
            prefix_hash=entry.prefix_hash,
            cache_type="kv",
            blocks=extracted_blocks,
            tokens=entry.tokens,  # Approximate
        )

    def remove(self, prefix_hash: str) -> bool:
        """
        Remove KV cache entry.

        Args:
            prefix_hash: Hash of entry to remove

        Returns:
            True if removed, False if not found
        """
        if prefix_hash in self._entries:
            del self._entries[prefix_hash]
            return True
        return False

    def clear(self) -> None:
        """
        Clear all KV cache entries.
        """
        self._entries.clear()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def stats(self) -> dict:
        """
        Get KV cache statistics.

        Returns:
            Dict with stats
        """
        total_memory = sum(e.size_bytes for e in self._entries.values())
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0.0

        return {
            "entries": len(self._entries),
            "max_entries": self.max_entries,
            "memory_bytes": total_memory,
            "memory_mb": total_memory / (1024 * 1024),
            "max_memory_mb": self.max_memory_mb,
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "hit_rate": hit_rate,
            "eviction_policy": self.eviction_policy,
        }

    def _maybe_evict(self) -> None:
        """
        Evict entries if capacity exceeded.
        """
        # Check entry count
        if len(self._entries) >= self.max_entries:
            self._evict_one()

        # Check memory budget
        if self.max_memory_mb is not None:
            current_mb = sum(e.size_bytes for e in self._entries.values()) / (1024 * 1024)
            if current_mb >= self.max_memory_mb:
                self._evict_one()

    def _evict_one(self) -> None:
        """
        Evict one entry based on policy.
        """
        if not self._entries:
            return

        if self.eviction_policy == "lru":
            # Evict least recently accessed
            victim_hash = min(
                self._entries.keys(),
                key=lambda h: self._entries[h].last_accessed,
            )
        elif self.eviction_policy == "lfu":
            # Evict least frequently accessed
            victim_hash = min(
                self._entries.keys(),
                key=lambda h: self._entries[h].access_count,
            )
        else:  # fifo
            # Evict oldest
            victim_hash = min(
                self._entries.keys(),
                key=lambda h: self._entries[h].created_at,
            )

        self.remove(victim_hash)
        self._evictions += 1
        logger.debug(f"[KVCacheBackend] Evicted {victim_hash} ({self.eviction_policy})")

    # -----------------------------------------------------------------
    # Block-based Cache Methods (optional)
    # -----------------------------------------------------------------

    def lookup_prefix_block(
        self,
        tokens: List[int],
        min_match_tokens: int = 16,
    ) -> tuple[List[int], Optional[List[Any]]]:
        """
        Look up shared prefix using BlockManager (if enabled).

        This is more efficient than the standard lookup when
        many requests share common prefixes (e.g., system prompts).

        Args:
            tokens: Token sequence to look up
            min_match_tokens: Minimum tokens for a match

        Returns:
            Tuple of (remaining_tokens, kv_state)
        """
        if not self.use_block_manager or self._prefix_integration is None:
            return tokens, None

        return self._prefix_integration.lookup_prefix(tokens, min_match_tokens)

    def store_prefix_block(
        self,
        request_id: str,
        prefix_tokens: List[int],
        kv_state: List[Any],
        prefix_hash: Optional[str] = None,
    ) -> Tuple[int, bool]:
        """
        Store prefix using BlockManager with CoW sharing (if enabled).

        This enables efficient prefix reuse across multiple requests.
        When a stored prefix is reused, the KV state is shared via
        reference counting. On first write to a shared block, CoW
        creates a copy.

        Args:
            request_id: Request ID for lifecycle tracking
            prefix_tokens: Prefix token sequence
            kv_state: KV cache state
            prefix_hash: Optional precomputed hash

        Returns:
            Tuple of (block_id, was_cache_hit)
        """
        if not self.use_block_manager or self._prefix_integration is None:
            # Fall back to standard store
            h = prefix_hash or self._compute_hash(prefix_tokens)
            entry = CacheEntry(
                entry_id=f"prefix_{h}",
                prefix_hash=h,
                cache_type="kv",
                blocks=kv_state,
                tokens=len(prefix_tokens),
            )
            self.store(h, entry)
            return -1, False

        return self._prefix_integration.store_prefix(
            request_id=request_id,
            prefix_tokens=prefix_tokens,
            kv_state=kv_state,
            prefix_hash=prefix_hash,
        )

    def release_prefix_blocks(
        self,
        request_id: str,
    ) -> None:
        """
        Release all shared prefix blocks for a request.

        Must be called when request completes to decrement
        reference counts and potentially free blocks.

        Args:
            request_id: Request ID to release
        """
        if not self.use_block_manager or self._prefix_integration is None:
            return

        self._prefix_integration.release_request(request_id)

    def copy_on_write_block(
        self,
        block_id: int,
        request_id: Optional[str] = None,
    ) -> int:
        """
        Perform Copy-on-Write for a shared block.

        If the block is shared (ref_count > 1), creates a copy.
        Otherwise, returns the same block ID.

        Args:
            block_id: Block ID to potentially copy
            request_id: Optional request ID for tracking

        Returns:
            New block ID (if copied) or same block_id
        """
        if not self.use_block_manager or self._cow is None:
            return block_id

        return self._cow.copy_on_write(block_id, request_id)

    def get_block_manager_stats(self) -> Optional[Dict[str, Any]]:
        """
        Get BlockManager statistics (if enabled).

        Returns:
            Dict with block stats or None if not enabled
        """
        if not self.use_block_manager or self._block_manager is None:
            return None

        stats = self._block_manager.get_stats()
        if self._cow is not None:
            stats["cow"] = self._cow.get_stats()
        return stats