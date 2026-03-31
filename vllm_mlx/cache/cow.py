# SPDX-License-Identifier: Apache-2.0
"""
CoWPrefixSharing - Copy-on-Write Prefix Sharing for vllm-mlx.

Enables efficient KV cache sharing across requests with common prefixes:

Flow:
1. Request arrives with prompt tokens
2. Compute prefix hash (e.g., system prompt + tools)
3. Check if prefix already cached
4. If cached:
   - Reuse existing block (ref_count++)
   - On first write: CoW -> new block for divergence
5. If not cached:
   - Allocate new block
   - Compute and cache KV state
   - Mark as shareable

Architecture:
    Shared Prefix Block
    ┌─────────────────────────────────────┐
    │ Shared Block (ref_count=3)          │
    │ tokens: [sys_prompt, tools, ...]    │
    │ kv_state: {keys, values, ...}       │
    └─────────────────────────────────────┘
              │         │         │
              ▼         ▼         ▼
    ┌─────────┐ ┌─────────┐ ┌─────────┐
    │ Request │ │ Request │ │ Request │
    │ A       │ │ B       │ │ C       │
    │ (copy)  │ │ (copy)  │ │ (copy)  │
    └─────────┘ └─────────┘ └─────────┘
    
    On Write (Request A diverges):
    ┌─────────────────────────────────────┐
    │ Shared Block (ref_count=2)          │
    └─────────────────────────────────────┘
              │         │
              ▼         ▼
    ┌─────────┐ ┌─────────┐
    │ Request │ │ Request │
    │ B, C    │ │ A       │
    └─────────┘ │ (new)   │
                └─────────┘
"""

import hashlib
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx

from .block_manager import BlockManager, KVBlock

logger = logging.getLogger(__name__)


@dataclass
class PrefixEntry:
    """
    Entry in the prefix cache.

    Attributes:
        prefix_hash: Hash of prefix tokens
        block_id: Block ID containing KV state
        tokens: Prefix token sequence
        ref_count: Number of requests using this prefix
        created_at: Creation timestamp
        last_accessed: Last access timestamp
    """

    prefix_hash: str
    block_id: int
    tokens: List[int]
    ref_count: int = 1
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)

    def update_access(self) -> None:
        """Update last accessed timestamp."""
        self.last_accessed = time.time()


class CoWPrefixSharing:
    """
    Copy-on-Write Prefix Sharing Manager.

    Manages shared prefix blocks with CoW semantics:
    - Multiple requests can share the same prefix block
    - When a request needs to modify a shared block, CoW creates a copy
    - Reference counting tracks sharing

    Integration with BlockManager:
        >>> block_manager = BlockManager(max_blocks=1000)
        >>> cow = CoWPrefixSharing(block_manager)
        >>> 
        >>> # Share prefix across requests
        >>> block_id = cow.share_prefix(prefix_hash, tokens, kv_state)
        >>> 
        >>> # When request diverges
        >>> new_block_id = cow.copy_on_write(block_id)

    Example Use Case - Multi-turn Chat:
        Request 1: [system] + [tools] + [user_msg_1]
        Request 2: [system] + [tools] + [user_msg_2]  # Shares prefix
        Request 3: [system] + [tools] + [user_msg_3]  # Shares prefix

        All three share the [system] + [tools] prefix block.
        Each request only computes KV for its user_msg.
    """

    def __init__(
        self,
        block_manager: BlockManager,
        enable_logging: bool = True,
    ):
        """
        Initialize CoW prefix sharing.

        Args:
            block_manager: BlockManager instance for block allocation
            enable_logging: Whether to log operations
        """
        self.block_manager = block_manager
        self.enable_logging = enable_logging

        # Shared prefix registry: hash -> PrefixEntry
        self._shared_prefixes: Dict[str, PrefixEntry] = {}

        # Block -> prefix hash mapping (for reverse lookup)
        self._block_to_hash: Dict[int, str] = {}

        # Block ref counts (managed here, synced with BlockManager)
        self._ref_counts: Dict[int, int] = {}

        # Statistics
        self._total_shares = 0
        self._total_cows = 0
        self._total_evictions = 0

        # Thread safety
        self._lock = threading.Lock()

        logger.info(
            f"[CoWPrefixSharing] Initialized with "
            f"block_manager(block_size={block_manager.block_size})"
        )

    def share_prefix(
        self,
        prefix_tokens: List[int],
        kv_state: Optional[List[Any]] = None,
        prefix_hash: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> Tuple[int, bool]:
        """
        Share a prefix block.

        If prefix already cached, returns existing block ID
        and increments ref count. Otherwise, creates new block.

        Args:
            prefix_tokens: Prefix token sequence
            kv_state: KV cache state for prefix
            prefix_hash: Optional precomputed hash
            request_id: Optional request ID for tracking

        Returns:
            Tuple of (block_id, was_cache_hit)
        """
        with self._lock:
            # Compute hash if not provided
            if prefix_hash is None:
                prefix_hash = self._compute_hash(prefix_tokens)

            # Check for existing shared prefix
            entry = self._shared_prefixes.get(prefix_hash)
            if entry is not None:
                # Cache hit - increment ref count
                entry.ref_count += 1
                entry.update_access()

                # Sync ref count with block manager
                block = self.block_manager.get_block(entry.block_id)
                if block is not None:
                    block.ref_count = entry.ref_count
                    block.update_access()

                self._total_shares += 1

                if self.enable_logging:
                    logger.debug(
                        f"[CoWPrefixSharing] Cache hit "
                        f"hash={prefix_hash[:8]} "
                        f"block={entry.block_id} "
                        f"ref_count={entry.ref_count} "
                        f"request={request_id[:12] if request_id else '?'}"
                    )

                return entry.block_id, True

            # Cache miss - create new shared block
            # Allocate block for prefix
            n_tokens = len(prefix_tokens)
            n_blocks_needed = (
                n_tokens + self.block_manager.block_size - 1
            ) // self.block_manager.block_size

            try:
                block_ids = self.block_manager.allocate(
                    request_id=f"prefix_{prefix_hash[:8]}",
                    n_tokens=n_tokens,
                )
            except ValueError as e:
                logger.warning(f"[CoWPrefixSharing] Cannot allocate prefix: {e}")
                return -1, False

            # Store KV state in first block
            if kv_state is not None and block_ids:
                self.block_manager.store_block(
                    block_ids[0],
                    prefix_tokens[:self.block_manager.block_size],
                    kv_state,
                )

            # Create entry
            block_id = block_ids[0]
            entry = PrefixEntry(
                prefix_hash=prefix_hash,
                block_id=block_id,
                tokens=prefix_tokens,
                ref_count=1,
            )

            self._shared_prefixes[prefix_hash] = entry
            self._block_to_hash[block_id] = prefix_hash
            self._ref_counts[block_id] = 1

            if self.enable_logging:
                logger.debug(
                    f"[CoWPrefixSharing] New prefix "
                    f"hash={prefix_hash[:8]} "
                    f"block={block_id} "
                    f"tokens={n_tokens} "
                    f"request={request_id[:12] if request_id else '?'}"
                )

            return block_id, False

    def copy_on_write(
        self,
        block_id: int,
        request_id: Optional[str] = None,
    ) -> int:
        """
        Perform Copy-on-Write for a shared block.

        If block has ref_count > 1, creates a copy.
        Otherwise, returns the same block.

        Args:
            block_id: Block ID to potentially copy
            request_id: Optional request ID for tracking

        Returns:
            New block ID (if copied) or same block_id
        """
        with self._lock:
            ref_count = self._ref_counts.get(block_id, 1)

            if ref_count <= 1:
                # Not shared, no need to copy
                return block_id

            # Get prefix hash for this block
            prefix_hash = self._block_to_hash.get(block_id)
            if prefix_hash is None:
                # Not a shared prefix block
                return self.block_manager.copy_on_write(block_id)

            # Get original entry
            entry = self._shared_prefixes.get(prefix_hash)
            if entry is None:
                return block_id

            # Create copy via block manager
            new_block_id = self.block_manager.copy_on_write(block_id)

            if new_block_id == block_id:
                # Copy failed
                return block_id

            # Update ref counts
            self._ref_counts[block_id] = ref_count - 1
            self._ref_counts[new_block_id] = 1
            entry.ref_count = ref_count - 1

            self._total_cows += 1

            if self.enable_logging:
                logger.info(
                    f"[CoWPrefixSharing] CoW "
                    f"block={block_id} -> {new_block_id} "
                    f"ref_count={ref_count} -> {ref_count - 1} "
                    f"request={request_id[:12] if request_id else '?'}"
                )

            return new_block_id

    def release_prefix(
        self,
        prefix_hash: str,
        request_id: Optional[str] = None,
    ) -> bool:
        """
        Release a shared prefix reference.

        Decrements ref count. If ref_count reaches 0,
        the block is freed.

        Args:
            prefix_hash: Hash of prefix to release
            request_id: Optional request ID for tracking

        Returns:
            True if released, False if not found
        """
        with self._lock:
            entry = self._shared_prefixes.get(prefix_hash)
            if entry is None:
                return False

            ref_count = entry.ref_count
            if ref_count <= 1:
                # Last reference - free the block
                block_id = entry.block_id
                self.block_manager.free(f"prefix_{prefix_hash[:8]}")

                del self._shared_prefixes[prefix_hash]
                if block_id in self._block_to_hash:
                    del self._block_to_hash[block_id]
                if block_id in self._ref_counts:
                    del self._ref_counts[block_id]

                self._total_evictions += 1

                if self.enable_logging:
                    logger.debug(
                        f"[CoWPrefixSharing] Released (freed) "
                        f"hash={prefix_hash[:8]} "
                        f"block={block_id} "
                        f"request={request_id[:12] if request_id else '?'}"
                    )
            else:
                # Decrement ref count
                entry.ref_count = ref_count - 1
                self._ref_counts[entry.block_id] = ref_count - 1

                if self.enable_logging:
                    logger.debug(
                        f"[CoWPrefixSharing] Released "
                        f"hash={prefix_hash[:8]} "
                        f"block={entry.block_id} "
                        f"ref_count={ref_count} -> {ref_count - 1} "
                        f"request={request_id[:12] if request_id else '?'}"
                    )

            return True

    def get_prefix_entry(
        self,
        prefix_hash: str,
    ) -> Optional[PrefixEntry]:
        """
        Get prefix entry by hash.

        Args:
            prefix_hash: Prefix hash

        Returns:
            PrefixEntry if found, None otherwise
        """
        return self._shared_prefixes.get(prefix_hash)

    def get_shared_prefix_stats(
        self,
        prefix_hash: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get statistics for a shared prefix.

        Args:
            prefix_hash: Prefix hash

        Returns:
            Dict with stats or None if not found
        """
        entry = self._shared_prefixes.get(prefix_hash)
        if entry is None:
            return None

        block = self.block_manager.get_block(entry.block_id)

        return {
            "prefix_hash": prefix_hash,
            "block_id": entry.block_id,
            "tokens": len(entry.tokens),
            "ref_count": entry.ref_count,
            "is_shared": block.is_shared if block else False,
            "created_at": entry.created_at,
            "last_accessed": entry.last_accessed,
        }

    def find_shared_prefix(
        self,
        tokens: List[int],
        min_match_tokens: int = 16,
    ) -> Tuple[Optional[str], int]:
        """
        Find if any prefix of the given tokens is shared.

        Args:
            tokens: Token sequence to check
            min_match_tokens: Minimum tokens for a match

        Returns:
            Tuple of (prefix_hash, num_matching_tokens) or (None, 0)
        """
        with self._lock:
            # Try progressively shorter prefixes
            for end_idx in range(len(tokens), min_match_tokens - 1, -1):
                prefix = tokens[:end_idx]
                prefix_hash = self._compute_hash(prefix)

                if prefix_hash in self._shared_prefixes:
                    return prefix_hash, end_idx

            return None, 0

    def _compute_hash(
        self,
        tokens: List[int],
    ) -> str:
        """
        Compute hash for token sequence.

        Args:
            tokens: Token IDs

        Returns:
            Hash string
        """
        token_bytes = b"".join(t.to_bytes(4, "little") for t in tokens)
        return hashlib.sha256(token_bytes).hexdigest()[:16]

    def get_stats(self) -> Dict[str, Any]:
        """
        Get CoW prefix sharing statistics.

        Returns:
            Dict with statistics
        """
        with self._lock:
            total_refs = sum(e.ref_count for e in self._shared_prefixes.values())
            total_tokens = sum(len(e.tokens) for e in self._shared_prefixes.values())

            return {
                "shared_prefixes": len(self._shared_prefixes),
                "total_references": total_refs,
                "total_tokens": total_tokens,
                "total_shares": self._total_shares,
                "total_cows": self._total_cows,
                "total_evictions": self._total_evictions,
                "block_manager_stats": self.block_manager.get_stats(),
            }

    def clear(self) -> None:
        """
        Clear all shared prefixes.
        """
        with self._lock:
            # Free all blocks
            for prefix_hash, entry in self._shared_prefixes.items():
                self.block_manager.free(f"prefix_{prefix_hash[:8]}")

            self._shared_prefixes.clear()
            self._block_to_hash.clear()
            self._ref_counts.clear()

            # Reset stats
            self._total_shares = 0
            self._total_cows = 0
            self._total_evictions = 0

            logger.info("[CoWPrefixSharing] Cleared all shared prefixes")


class PrefixSharingIntegration:
    """
    Integration layer for CoWPrefixSharing with KVCacheBackend.

    Provides seamless integration between the existing cache
    system and the new CoW prefix sharing.

    Usage:
        >>> integration = PrefixSharingIntegration(block_manager)
        >>> 
        >>> # Store prefix
        >>> integration.store_prefix(request_id, prefix_tokens, kv_state)
        >>> 
        >>> # Lookup prefix
        >>> cached_tokens, kv_state = integration.lookup_prefix(tokens)
        >>> 
        >>> # Release when request completes
        >>> integration.release_request(request_id)
    """

    def __init__(
        self,
        block_manager: BlockManager,
    ):
        """
        Initialize integration layer.

        Args:
            block_manager: BlockManager instance
        """
        self.block_manager = block_manager
        self.cow = CoWPrefixSharing(block_manager)

        # Request -> shared prefixes mapping
        self._request_prefixes: Dict[str, List[str]] = {}

    def store_prefix(
        self,
        request_id: str,
        prefix_tokens: List[int],
        kv_state: List[Any],
        prefix_hash: Optional[str] = None,
    ) -> Tuple[int, bool]:
        """
        Store prefix for future sharing.

        Args:
            request_id: Request ID
            prefix_tokens: Prefix tokens
            kv_state: KV cache state
            prefix_hash: Optional precomputed hash

        Returns:
            Tuple of (block_id, was_cache_hit)
        """
        block_id, cache_hit = self.cow.share_prefix(
            prefix_tokens=prefix_tokens,
            kv_state=kv_state,
            prefix_hash=prefix_hash,
            request_id=request_id,
        )

        # Track for request lifecycle
        if prefix_hash is None:
            prefix_hash = self.cow._compute_hash(prefix_tokens)

        if request_id not in self._request_prefixes:
            self._request_prefixes[request_id] = []
        self._request_prefixes[request_id].append(prefix_hash)

        return block_id, cache_hit

    def lookup_prefix(
        self,
        tokens: List[int],
        min_match_tokens: int = 16,
    ) -> Tuple[List[int], Optional[List[Any]]]:
        """
        Look up shared prefix for given tokens.

        Args:
            tokens: Token sequence
            min_match_tokens: Minimum match length

        Returns:
            Tuple of (remaining_tokens, kv_state)
        """
        prefix_hash, match_len = self.cow.find_shared_prefix(
            tokens, min_match_tokens
        )

        if prefix_hash is None:
            return tokens, None

        entry = self.cow.get_prefix_entry(prefix_hash)
        if entry is None:
            return tokens, None

        # Get KV state from block
        block = self.block_manager.get_block(entry.block_id)
        if block is None or block.kv_state is None:
            return tokens, None

        remaining = tokens[match_len:]

        logger.debug(
            f"[PrefixSharingIntegration] Lookup hit "
            f"hash={prefix_hash[:8]} "
            f"match={match_len} "
            f"remaining={len(remaining)}"
        )

        return remaining, block.kv_state

    def release_request(
        self,
        request_id: str,
    ) -> None:
        """
        Release all shared prefixes for a request.

        Args:
            request_id: Request ID to release
        """
        prefixes = self._request_prefixes.pop(request_id, [])
        for prefix_hash in prefixes:
            self.cow.release_prefix(prefix_hash, request_id)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get integration statistics.

        Returns:
            Dict with stats
        """
        return {
            "active_requests": len(self._request_prefixes),
            "cow_stats": self.cow.get_stats(),
        }