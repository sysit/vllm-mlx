# SPDX-License-Identifier: Apache-2.0
"""
Base CacheBackend interface for unified cache management.

This module provides the abstract base class for all cache backend types,
ensuring a consistent interface across KV cache, vision cache, and hybrid cache.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, List


@dataclass
class CacheEntry:
    """
    Unified cache entry representation.

    A cache entry represents cached state that can be reused across requests.
    Different cache types store different kinds of state:
    - KV cache: Key-Value pairs from transformer attention layers
    - Vision cache: Pre-computed vision embeddings from image processing
    - Hybrid cache: Both KV and RNN (Mamba/ArraysCache) states

    Attributes:
        entry_id: Unique identifier for this cache entry
        prefix_hash: Hash of the prefix tokens/content that generated this cache
        cache_type: Type identifier ("kv", "vision", "hybrid")
        blocks: List of cache block objects (actual cached state)
        vision_embeddings: Optional vision embeddings (for MLLM)
        rnn_snapshots: Optional RNN state snapshots (for hybrid models)
        tokens: Number of tokens this cache covers
        created_at: Timestamp when entry was created
        last_accessed: Timestamp of last access
        access_count: Number of times this entry has been accessed
    """

    entry_id: str
    prefix_hash: str
    cache_type: str  # "kv", "vision", "hybrid"
    blocks: List[Any] = field(default_factory=list)
    vision_embeddings: Optional[Any] = None
    rnn_snapshots: Optional[dict] = None
    tokens: int = 0
    created_at: float = 0.0
    last_accessed: float = 0.0
    access_count: int = 0

    @property
    def size_bytes(self) -> int:
        """
        Estimate memory size of this cache entry in bytes.

        Returns:
            Estimated memory footprint
        """
        total = 0

        # Estimate KV cache size
        for block in self.blocks:
            if block is None:
                continue
            # KVCache: keys + values arrays
            if hasattr(block, "keys") and block.keys is not None:
                total += block.keys.nbytes
            if hasattr(block, "values") and block.values is not None:
                total += block.values.nbytes
            # ArraysCache/MambaCache: state arrays
            if hasattr(block, "cache") and isinstance(block.cache, list):
                for arr in block.cache:
                    if arr is not None:
                        total += arr.nbytes

        # Vision embeddings size
        if self.vision_embeddings is not None:
            if hasattr(self.vision_embeddings, "nbytes"):
                total += self.vision_embeddings.nbytes
            elif isinstance(self.vision_embeddings, dict):
                for v in self.vision_embeddings.values():
                    if hasattr(v, "nbytes"):
                        total += v.nbytes

        return total

    def is_valid(self) -> bool:
        """
        Check if this cache entry is still valid and usable.

        Returns:
            True if cache can be safely used
        """
        if not self.blocks:
            return False

        for block in self.blocks:
            if block is None:
                return False
            # Check KVCache validity
            if hasattr(block, "keys") and block.keys is None:
                return False
            if hasattr(block, "values") and block.values is None:
                return False
            # Check batch dimension (must be 1 for single-sequence entries)
            if hasattr(block, "keys") and block.keys is not None:
                if block.keys.shape[0] != 1:
                    return False

        return True

    def update_access(self, timestamp: float) -> None:
        """
        Update access statistics.

        Args:
            timestamp: Current timestamp
        """
        self.last_accessed = timestamp
        self.access_count += 1


class CacheBackend(ABC):
    """
    Abstract base class for cache backends.

    Provides a unified interface for different cache types:
    - KVCacheBackend: Standard transformer KV cache
    - VisionCacheBackend: Vision embeddings cache (no batch)
    - HybridCacheBackend: Mamba+Transformer hybrid cache

    Each backend handles its specific cache type while providing
    consistent lookup/store/merge semantics.
    """

    @abstractmethod
    def lookup(self, prefix_hash: str) -> Optional[CacheEntry]:
        """
        Look up a cache entry by prefix hash.

        Args:
            prefix_hash: Hash of the prefix content

        Returns:
            CacheEntry if found, None otherwise
        """
        pass

    @abstractmethod
    def store(self, prefix_hash: str, entry: CacheEntry) -> None:
        """
        Store a cache entry.

        Args:
            prefix_hash: Hash of the prefix content
            entry: Cache entry to store
        """
        pass

    @abstractmethod
    def supports_batch(self) -> bool:
        """
        Check if this backend supports batch operations.

        Returns:
            True if batch merge/extract is supported
        """
        pass

    @abstractmethod
    def merge(self, entries: List[CacheEntry]) -> CacheEntry:
        """
        Merge multiple cache entries into a batched entry.

        Args:
            entries: List of cache entries to merge

        Returns:
            New batched CacheEntry

        Raises:
            NotImplementedError: If backend doesn't support batching
        """
        pass

    @abstractmethod
    def extract(self, entry: CacheEntry, idx: int) -> CacheEntry:
        """
        Extract a single entry from a batched entry.

        Args:
            entry: Batched cache entry
            idx: Index to extract

        Returns:
            New single-sequence CacheEntry

        Raises:
            NotImplementedError: If backend doesn't support batching
        """
        pass

    @abstractmethod
    def remove(self, prefix_hash: str) -> bool:
        """
        Remove a cache entry.

        Args:
            prefix_hash: Hash of the entry to remove

        Returns:
            True if entry was removed, False if not found
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """
        Clear all cache entries.
        """
        pass

    @abstractmethod
    def stats(self) -> dict:
        """
        Get cache statistics.

        Returns:
            Dict with entries count, memory usage, hit rate, etc.
        """
        pass

    def get_or_compute(
        self,
        prefix_hash: str,
        compute_fn: callable,
    ) -> CacheEntry:
        """
        Get existing entry or compute new one.

        Args:
            prefix_hash: Hash to lookup
            compute_fn: Function to compute new entry if not found

        Returns:
            Existing or newly computed CacheEntry
        """
        entry = self.lookup(prefix_hash)
        if entry is not None:
            return entry

        entry = compute_fn()
        self.store(prefix_hash, entry)
        return entry