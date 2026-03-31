# SPDX-License-Identifier: Apache-2.0
"""
BlockManager - Block-based KV Cache Management for vllm-mlx.

Borrowing from vLLM's PagedAttention design:
- Fixed-size blocks for KV cache allocation
- Block tables for request -> block mapping
- Efficient memory management with block reuse
- Support for prefix sharing via CoW

Architecture:
    KV Cache Layout:
    ┌────────┬────────┬────────┬────────┐
    │ Block 0│ Block 1│ Block 2│ Block 3│  ...
    └────────┴────────┴────────┴────────┘
    
    Request Mapping:
    request_id -> [block_id_0, block_id_1, ...]
    
    Block Structure:
    Block {
        block_id: int
        tokens: [token_0, token_1, ..., token_15]  # BLOCK_SIZE tokens
        kv_state: {keys: (1, layers, 16, dim), values: ...}
        ref_count: int  # For CoW sharing
        is_shared: bool  # Shared prefix block?
    }
"""

import hashlib
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx

logger = logging.getLogger(__name__)


# Default block size (tokens per block)
# Smaller blocks = finer granularity for prefix sharing
# Larger blocks = less overhead, better memory locality
DEFAULT_BLOCK_SIZE = 16


@dataclass
class KVBlock:
    """
    A single KV cache block.

    Attributes:
        block_id: Unique block identifier
        tokens: List of token IDs in this block
        kv_state: Cached KV tensors (or Mamba state for hybrid models)
        ref_count: Reference count for CoW
        is_shared: Whether this block is shared across requests
        created_at: Creation timestamp
        last_accessed: Last access timestamp
    """

    block_id: int
    tokens: List[int] = field(default_factory=list)
    kv_state: Optional[List[Any]] = None
    ref_count: int = 1
    is_shared: bool = False
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)

    def is_full(self) -> bool:
        """Check if block is full."""
        return len(self.tokens) >= DEFAULT_BLOCK_SIZE

    def num_tokens(self) -> int:
        """Get number of tokens in block."""
        return len(self.tokens)

    def size_bytes(self) -> int:
        """Estimate memory size of block."""
        if self.kv_state is None:
            return 0

        total = 0
        for layer_state in self.kv_state:
            if isinstance(layer_state, dict):
                # Extracted state format
                keys = layer_state.get("keys")
                values = layer_state.get("values")
                if keys is not None:
                    total += keys.nbytes
                if values is not None:
                    total += values.nbytes
            elif hasattr(layer_state, "keys"):
                # KVCache object
                if layer_state.keys is not None:
                    total += layer_state.keys.nbytes
                if layer_state.values is not None:
                    total += layer_state.values.nbytes

        return total

    def update_access(self) -> None:
        """Update last accessed timestamp."""
        self.last_accessed = time.time()


@dataclass
class BlockTable:
    """
    Block table for a single request.

    Maps request to its allocated blocks and tracks:
    - Physical block IDs
    - Token coverage
    - Shared prefix blocks vs. exclusive blocks
    """

    request_id: str
    block_ids: List[int] = field(default_factory=list)
    num_tokens: int = 0
    shared_prefix_blocks: int = 0  # Number of blocks shared with other requests

    def add_block(self, block_id: int) -> None:
        """Add a block to the table."""
        self.block_ids.append(block_id)

    def remove_block(self, block_id: int) -> bool:
        """Remove a block from the table."""
        if block_id in self.block_ids:
            self.block_ids.remove(block_id)
            return True
        return False

    def get_blocks(self) -> List[int]:
        """Get all block IDs."""
        return list(self.block_ids)


class BlockManager:
    """
    Block-based KV Cache Manager.

    Features:
    - Fixed-size block allocation
    - Efficient prefix sharing
    - Reference counting for CoW
    - LRU eviction for memory management
    - Integration with existing prefix cache

    Example:
        >>> manager = BlockManager(max_blocks=1000)
        >>> blocks = manager.allocate("req_1", 100)  # Allocate for 100 tokens
        >>> manager.store_block(blocks[0], tokens[:16], kv_state)
        >>> manager.share_prefix("req_2", prefix_tokens)  # Share prefix with req_2
    """

    def __init__(
        self,
        max_blocks: int = 1000,
        block_size: int = DEFAULT_BLOCK_SIZE,
        max_memory_mb: Optional[int] = None,
    ):
        """
        Initialize block manager.

        Args:
            max_blocks: Maximum number of blocks
            block_size: Tokens per block (default: 16)
            max_memory_mb: Memory budget in MB (None = unlimited)
        """
        self.max_blocks = max_blocks
        self.block_size = block_size
        self.max_memory_mb = max_memory_mb

        # Block storage
        self._blocks: Dict[int, KVBlock] = {}
        self._free_blocks: List[int] = []  # Available block IDs
        self._next_block_id: int = 0

        # Request -> blocks mapping
        self._block_tables: Dict[str, BlockTable] = {}

        # Prefix hash -> shared block mapping
        self._shared_prefixes: Dict[str, int] = {}  # hash -> block_id

        # Reference counts for CoW
        self._ref_counts: Dict[int, int] = {}

        # Statistics
        self._total_blocks_allocated = 0
        self._total_blocks_freed = 0
        self._cache_hits = 0
        self._cache_misses = 0

        # Thread safety lock
        self._lock = threading.Lock()

        # Pre-allocate block IDs
        for i in range(max_blocks):
            self._free_blocks.append(i)

        logger.info(
            f"[BlockManager] Initialized with "
            f"max_blocks={max_blocks}, block_size={block_size}"
        )

    def allocate(
        self,
        request_id: str,
        n_tokens: int,
    ) -> List[int]:
        """
        Allocate blocks for a request.

        Args:
            request_id: Request identifier
            n_tokens: Number of tokens to allocate for

        Returns:
            List of allocated block IDs

        Raises:
            ValueError: If not enough blocks available
        """
        with self._lock:
            # Calculate number of blocks needed
            n_blocks_needed = (n_tokens + self.block_size - 1) // self.block_size

            # Check availability
            if len(self._free_blocks) < n_blocks_needed:
                # Try to evict
                self._evict_blocks(n_blocks_needed - len(self._free_blocks))

                if len(self._free_blocks) < n_blocks_needed:
                    raise ValueError(
                        f"Not enough blocks: need {n_blocks_needed}, "
                        f"available {len(self._free_blocks)}"
                    )

            # Allocate blocks
            allocated = []
            for _ in range(n_blocks_needed):
                block_id = self._free_blocks.pop()
                block = KVBlock(block_id=block_id)
                self._blocks[block_id] = block
                self._ref_counts[block_id] = 1
                allocated.append(block_id)

            # Create block table
            table = BlockTable(request_id=request_id, block_ids=allocated)
            table.num_tokens = n_tokens
            self._block_tables[request_id] = table

            self._total_blocks_allocated += n_blocks_needed

            logger.debug(
                f"[BlockManager] Allocated {n_blocks_needed} blocks "
                f"for request {request_id[:12]}"
            )

            return allocated

    def free(
        self,
        request_id: str,
    ) -> None:
        """
        Free all blocks for a request.

        Handles CoW: decrements ref counts, only frees
        blocks with ref_count == 0.

        Args:
            request_id: Request to free
        """
        with self._lock:
            table = self._block_tables.get(request_id)
            if table is None:
                return

            for block_id in table.block_ids:
                ref_count = self._ref_counts.get(block_id, 0)
                if ref_count > 1:
                    # Shared block - decrement ref count
                    self._ref_counts[block_id] = ref_count - 1
                else:
                    # Exclusive block - free it
                    if block_id in self._blocks:
                        del self._blocks[block_id]
                    if block_id in self._ref_counts:
                        del self._ref_counts[block_id]
                    self._free_blocks.append(block_id)
                    self._total_blocks_freed += 1

            del self._block_tables[request_id]

            logger.debug(
                f"[BlockManager] Freed blocks for request {request_id[:12]}"
            )

    def store_block(
        self,
        block_id: int,
        tokens: List[int],
        kv_state: Optional[List[Any]] = None,
    ) -> None:
        """
        Store tokens and KV state in a block.

        Args:
            block_id: Block ID to store in
            tokens: Token IDs (up to block_size)
            kv_state: Optional KV cache state
        """
        with self._lock:
            block = self._blocks.get(block_id)
            if block is None:
                logger.warning(f"[BlockManager] Unknown block {block_id}")
                return

            # Truncate tokens to block_size
            block.tokens = tokens[:self.block_size]
            block.kv_state = kv_state
            block.update_access()

    def get_block(self, block_id: int) -> Optional[KVBlock]:
        """
        Get a block by ID.

        Args:
            block_id: Block ID

        Returns:
            KVBlock if exists, None otherwise
        """
        return self._blocks.get(block_id)

    def get_block_table(self, request_id: str) -> Optional[BlockTable]:
        """
        Get block table for a request.

        Args:
            request_id: Request ID

        Returns:
            BlockTable if exists, None otherwise
        """
        return self._block_tables.get(request_id)

    def can_share_prefix(
        self,
        prefix_hash: str,
    ) -> bool:
        """
        Check if a prefix can be shared.

        Args:
            prefix_hash: Hash of prefix tokens

        Returns:
            True if prefix has shared blocks
        """
        return prefix_hash in self._shared_prefixes

    def share_prefix(
        self,
        request_id: str,
        prefix_tokens: List[int],
        prefix_hash: Optional[str] = None,
    ) -> List[int]:
        """
        Share prefix blocks with a request.

        If prefix already cached, reuse existing blocks.
        Otherwise, allocate new blocks and mark as shared.

        Args:
            request_id: Target request ID
            prefix_tokens: Prefix token IDs
            prefix_hash: Optional precomputed hash

        Returns:
            List of shared block IDs
        """
        with self._lock:
            if prefix_hash is None:
                prefix_hash = self._compute_hash(prefix_tokens)

            # Check if prefix already cached
            if prefix_hash in self._shared_prefixes:
                # Reuse existing blocks
                shared_block_id = self._shared_prefixes[prefix_hash]
                block = self._blocks.get(shared_block_id)
                if block is None:
                    # Block was evicted, remove hash
                    del self._shared_prefixes[prefix_hash]
                    self._cache_misses += 1
                    return self._allocate_new_shared_prefix(
                        request_id, prefix_tokens, prefix_hash
                    )

                # Increment ref count
                self._ref_counts[shared_block_id] = (
                    self._ref_counts.get(shared_block_id, 1) + 1
                )
                block.is_shared = True
                block.update_access()

                # Create block table with shared block
                table = BlockTable(request_id=request_id)
                table.add_block(shared_block_id)
                table.shared_prefix_blocks = 1
                table.num_tokens = len(prefix_tokens)
                self._block_tables[request_id] = table

                self._cache_hits += 1
                logger.debug(
                    f"[BlockManager] Prefix cache hit "
                    f"hash={prefix_hash[:8]} "
                    f"request={request_id[:12]}"
                )

                return [shared_block_id]

            # Allocate new shared prefix
            return self._allocate_new_shared_prefix(
                request_id, prefix_tokens, prefix_hash
            )

    def _allocate_new_shared_prefix(
        self,
        request_id: str,
        prefix_tokens: List[int],
        prefix_hash: str,
    ) -> List[int]:
        """Allocate new blocks for a shared prefix."""
        n_blocks = (len(prefix_tokens) + self.block_size - 1) // self.block_size

        if len(self._free_blocks) < n_blocks:
            self._evict_blocks(n_blocks - len(self._free_blocks))
            if len(self._free_blocks) < n_blocks:
                self._cache_misses += 1
                return []

        allocated = []
        for i in range(n_blocks):
            block_id = self._free_blocks.pop()
            start_idx = i * self.block_size
            end_idx = min(start_idx + self.block_size, len(prefix_tokens))

            block = KVBlock(
                block_id=block_id,
                tokens=prefix_tokens[start_idx:end_idx],
                is_shared=True,
            )
            self._blocks[block_id] = block
            self._ref_counts[block_id] = 1
            allocated.append(block_id)

        # Register as shared prefix
        self._shared_prefixes[prefix_hash] = allocated[0]
        self._total_blocks_allocated += n_blocks

        # Create block table
        table = BlockTable(request_id=request_id, block_ids=allocated)
        table.shared_prefix_blocks = n_blocks
        table.num_tokens = len(prefix_tokens)
        self._block_tables[request_id] = table

        self._cache_misses += 1
        logger.debug(
            f"[BlockManager] Allocated new shared prefix "
            f"hash={prefix_hash[:8]} "
            f"blocks={n_blocks} "
            f"request={request_id[:12]}"
        )

        return allocated

    def copy_on_write(
        self,
        block_id: int,
    ) -> int:
        """
        Perform Copy-on-Write for a shared block.

        If block has ref_count > 1, creates a copy.
        Otherwise, returns the same block.

        Args:
            block_id: Block ID to potentially copy

        Returns:
            New block ID (if copied) or same block_id
        """
        with self._lock:
            ref_count = self._ref_counts.get(block_id, 1)

            if ref_count <= 1:
                # Not shared, no need to copy
                return block_id

            # Need to copy
            if not self._free_blocks:
                self._evict_blocks(1)
                if not self._free_blocks:
                    logger.warning("[BlockManager] No blocks for CoW")
                    return block_id

            # Create new block
            new_block_id = self._free_blocks.pop()
            old_block = self._blocks.get(block_id)

            if old_block is None:
                return block_id

            # Copy block data
            new_block = KVBlock(
                block_id=new_block_id,
                tokens=list(old_block.tokens),
                kv_state=self._copy_kv_state(old_block.kv_state),
                is_shared=False,
            )
            self._blocks[new_block_id] = new_block
            self._ref_counts[new_block_id] = 1

            # Decrement old block ref count
            self._ref_counts[block_id] = ref_count - 1

            self._total_blocks_allocated += 1

            logger.debug(
                f"[BlockManager] CoW: copied block {block_id} -> {new_block_id}"
            )

            return new_block_id

    def _copy_kv_state(
        self,
        kv_state: Optional[List[Any]],
    ) -> Optional[List[Any]]:
        """
        Copy KV state tensors.

        Args:
            kv_state: Original KV state

        Returns:
            Copied KV state
        """
        if kv_state is None:
            return None

        copied = []
        for layer_state in kv_state:
            if isinstance(layer_state, dict):
                # Extracted state format - copy arrays
                copied_layer = {}
                for key, value in layer_state.items():
                    if isinstance(value, mx.array):
                        copied_layer[key] = value.copy()
                    else:
                        copied_layer[key] = value
                copied.append(copied_layer)
            elif hasattr(layer_state, "state"):
                # KVCache object - use from_state
                try:
                    state = layer_state.state
                    meta = layer_state.meta_state if hasattr(layer_state, "meta_state") else None
                    cls = type(layer_state)
                    if hasattr(cls, "from_state"):
                        copied.append(cls.from_state(state, meta))
                    else:
                        copied.append(layer_state)
                except Exception as e:
                    logger.debug(f"[BlockManager] Failed to copy KV state: {e}")
                    copied.append(layer_state)
            else:
                copied.append(layer_state)

        return copied

    def _compute_hash(
        self,
        tokens: List[int],
    ) -> str:
        """
        Compute hash for a sequence of tokens.

        Args:
            tokens: Token IDs

        Returns:
            Hash string
        """
        token_bytes = b"".join(t.to_bytes(4, "little") for t in tokens)
        return hashlib.sha256(token_bytes).hexdigest()[:16]

    def _evict_blocks(
        self,
        n_needed: int,
    ) -> None:
        """
        Evict blocks to free memory.

        Uses LRU policy: evicts least recently accessed blocks
        with ref_count == 1 (exclusive, not shared).

        Args:
            n_needed: Number of blocks needed
        """
        # Find evictable blocks (ref_count == 1, not in active tables)
        evictable = []
        for block_id, block in self._blocks.items():
            ref_count = self._ref_counts.get(block_id, 1)
            if ref_count == 1:
                # Check if block is in any active table
                in_use = False
                for table in self._block_tables.values():
                    if block_id in table.block_ids:
                        in_use = True
                        break
                if not in_use:
                    evictable.append((block_id, block.last_accessed))

        # Sort by last accessed (oldest first)
        evictable.sort(key=lambda x: x[1])

        # Evict
        n_evicted = 0
        for block_id, _ in evictable:
            if n_evicted >= n_needed:
                break

            # Remove shared prefix hash if applicable
            for hash_key, shared_id in list(self._shared_prefixes.items()):
                if shared_id == block_id:
                    del self._shared_prefixes[hash_key]

            del self._blocks[block_id]
            if block_id in self._ref_counts:
                del self._ref_counts[block_id]
            self._free_blocks.append(block_id)
            n_evicted += 1
            self._total_blocks_freed += 1

        if n_evicted > 0:
            logger.debug(f"[BlockManager] Evicted {n_evicted} blocks (LRU)")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get block manager statistics.

        Returns:
            Dict with stats
        """
        with self._lock:
            total_memory = sum(b.size_bytes() for b in self._blocks.values())
            shared_blocks = sum(1 for b in self._blocks.values() if b.is_shared)

            return {
                "total_blocks": len(self._blocks),
                "free_blocks": len(self._free_blocks),
                "max_blocks": self.max_blocks,
                "block_size": self.block_size,
                "active_requests": len(self._block_tables),
                "shared_blocks": shared_blocks,
                "shared_prefixes": len(self._shared_prefixes),
                "memory_bytes": total_memory,
                "memory_mb": total_memory / (1024 * 1024),
                "cache_hits": self._cache_hits,
                "cache_misses": self._cache_misses,
                "hit_rate": self._cache_hits / (self._cache_hits + self._cache_misses)
                if (self._cache_hits + self._cache_misses) > 0
                else 0.0,
                "total_allocated": self._total_blocks_allocated,
                "total_freed": self._total_blocks_freed,
            }

    def clear(self) -> None:
        """
        Clear all blocks and tables.
        """
        with self._lock:
            self._blocks.clear()
            self._block_tables.clear()
            self._shared_prefixes.clear()
            self._ref_counts.clear()

            # Reset free blocks
            self._free_blocks = list(range(self.max_blocks))
            self._next_block_id = self.max_blocks

            # Reset stats
            self._total_blocks_allocated = 0
            self._total_blocks_freed = 0
            self._cache_hits = 0
            self._cache_misses = 0

            logger.info("[BlockManager] Cleared all blocks")