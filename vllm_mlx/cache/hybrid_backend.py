# SPDX-License-Identifier: Apache-2.0
"""
HybridCacheBackend - Cache backend for Mamba+Transformer hybrid models.

Hybrid models like Qwen3-Next mix attention layers (KVCache) with
recurrent layers (MambaCache/ArraysCache). This backend handles
the unique challenges of caching such hybrid architectures:

1. RNN state pollution: Rejected MTP drafts permanently pollute
   MambaCache state, requiring snapshot/restore mechanisms
2. Mixed cache types: Some layers are trimmable (KVCache),
   others are cumulative (ArraysCache)
3. Batch operations: Hybrid caches can be batched but require
   special handling for RNN state

This backend solves P0-001 (cache pollution) by providing
snapshot_rnn_state() and restore_rnn_state() methods.
"""

import logging
import time
from typing import Any, Optional, List, Dict

import mlx.core as mx

from .backend import CacheBackend, CacheEntry

logger = logging.getLogger(__name__)


class HybridCacheBackend(CacheBackend):
    """
    Cache backend for hybrid Mamba+Transformer models.

    Hybrid models present unique caching challenges:
    - KVCache layers: Support trim() to undo speculative tokens
    - ArraysCache/MambaCache layers: Cumulative state, irreversible

    When MTP (Multi-Token Prediction) rejects a draft token:
    - KVCache can trim(1) to undo the draft
    - ArraysCache state is polluted permanently

    This backend provides snapshot/restore to handle rejection:
    1. Before verify: snapshot_rnn_state() captures RNN state
    2. On reject: trim KV, restore RNN, re-advance with primary only

    This ensures both cache types end up consistent at [..., primary].
    """

    def __init__(
        self,
        max_entries: int = 100,
        max_memory_mb: Optional[int] = None,
        eviction_policy: str = "lru",
    ):
        """
        Initialize hybrid cache backend.

        Args:
            max_entries: Maximum cache entries
            max_memory_mb: Maximum memory budget
            eviction_policy: Eviction policy
        """
        self.max_entries = max_entries
        self.max_memory_mb = max_memory_mb
        self.eviction_policy = eviction_policy

        self._entries: Dict[str, CacheEntry] = {}
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._rnn_snapshots: Dict[str, dict] = {}  # Active snapshots

    def is_hybrid_cache(self, cache: List[Any]) -> bool:
        """
        Check if a cache list contains hybrid (ArraysCache + KVCache) layers.

        Args:
            cache: List of cache layer objects

        Returns:
            True if cache has both trimmable and non-trimmable layers
        """
        has_trimmable = False
        has_non_trimmable = False

        for layer_cache in cache:
            if hasattr(layer_cache, "is_trimmable"):
                if layer_cache.is_trimmable():
                    has_trimmable = True
                else:
                    has_non_trimmable = True
            elif hasattr(layer_cache, "state"):
                # ArraysCache/MambaCache has state, not trimmable
                has_non_trimmable = True
            elif hasattr(layer_cache, "keys"):
                # KVCache is trimmable
                has_trimmable = True

        return has_trimmable and has_non_trimmable

    def snapshot_rnn_state(self, cache: List[Any], snapshot_id: str) -> dict:
        """
        Snapshot RNN (ArraysCache/MambaCache) state before speculative verify.

        Args:
            cache: List of cache layers
            snapshot_id: Unique ID for this snapshot

        Returns:
            Dict mapping layer index to state arrays
        """
        snapshots = {}

        for layer_idx, layer_cache in enumerate(cache):
            # Check if layer is non-trimmable (RNN-like)
            is_rnn = False
            if hasattr(layer_cache, "is_trimmable"):
                is_rnn = not layer_cache.is_trimmable()
            elif hasattr(layer_cache, "state") and not hasattr(layer_cache, "keys"):
                is_rnn = True

            if is_rnn and hasattr(layer_cache, "state"):
                # Snapshot state arrays
                snapshots[layer_idx] = [
                    s.copy() if s is not None else None
                    for s in layer_cache.state
                ]

        self._rnn_snapshots[snapshot_id] = snapshots
        logger.debug(
            f"[HybridCacheBackend] Snapshot {snapshot_id}: "
            f"{len(snapshots)} RNN layers"
        )
        return snapshots

    def restore_rnn_state(self, cache: List[Any], snapshot_id: str) -> bool:
        """
        Restore RNN state from snapshot after rejected draft.

        Args:
            cache: List of cache layers
            snapshot_id: ID of snapshot to restore

        Returns:
            True if restore succeeded
        """
        snapshots = self._rnn_snapshots.get(snapshot_id)
        if snapshots is None:
            logger.warning(f"[HybridCacheBackend] Snapshot {snapshot_id} not found")
            return False

        for layer_idx, state_arrays in snapshots.items():
            if layer_idx < len(cache):
                cache[layer_idx].state = state_arrays

        logger.debug(f"[HybridCacheBackend] Restored {snapshot_id}")
        return True

    def discard_snapshot(self, snapshot_id: str) -> None:
        """
        Discard snapshot after successful verification (no restore needed).

        Args:
            snapshot_id: ID of snapshot to discard
        """
        if snapshot_id in self._rnn_snapshots:
            del self._rnn_snapshots[snapshot_id]

    def handle_mtp_reject(
        self,
        cache: List[Any],
        snapshot_id: str,
        trim_count: int = 1,
    ) -> None:
        """
        Handle MTP rejection: trim KV, restore RNN.

        This is the core logic for P0-001 fix:
        1. Trim trimmable (KV) layers by trim_count
        2. Restore RNN layers from snapshot
        3. Discard snapshot

        Args:
            cache: List of cache layers
            snapshot_id: Snapshot to restore
            trim_count: How many tokens to trim from KV layers
        """
        # Trim KV layers
        for layer_cache in cache:
            if hasattr(layer_cache, "is_trimmable") and layer_cache.is_trimmable():
                try:
                    layer_cache.trim(trim_count)
                except Exception as e:
                    logger.warning(f"[HybridCacheBackend] Trim failed: {e}")

        # Restore RNN layers
        self.restore_rnn_state(cache, snapshot_id)

        # Discard snapshot
        self.discard_snapshot(snapshot_id)

        logger.debug(
            f"[HybridCacheBackend] MTP reject handled: "
            f"trimmed {trim_count}, restored RNN"
        )

    def lookup(self, prefix_hash: str) -> Optional[CacheEntry]:
        """
        Look up hybrid cache entry.

        Args:
            prefix_hash: Hash of prefix tokens

        Returns:
            CacheEntry if found and valid
        """
        entry = self._entries.get(prefix_hash)
        if entry is None:
            self._misses += 1
            return None

        # Validate entry
        if not entry.is_valid():
            self.remove(prefix_hash)
            self._misses += 1
            return None

        # Check for stored RNN snapshots
        if entry.rnn_snapshots is not None:
            # Restore snapshots to active dict
            for snap_id, snap_data in entry.rnn_snapshots.items():
                self._rnn_snapshots[snap_id] = snap_data

        entry.update_access(time.time())
        self._hits += 1
        return entry

    def store(self, prefix_hash: str, entry: CacheEntry) -> None:
        """
        Store hybrid cache entry.

        Args:
            prefix_hash: Hash of prefix tokens
            entry: Cache entry with hybrid cache blocks
        """
        if not entry.blocks:
            logger.warning("[HybridCacheBackend] Cannot store empty entry")
            return

        # Mark as hybrid type
        entry.cache_type = "hybrid"

        # Store any active snapshots with the entry
        if self._rnn_snapshots:
            entry.rnn_snapshots = dict(self._rnn_snapshots)

        # Check capacity
        self._maybe_evict()

        # Store entry
        entry.created_at = time.time()
        entry.last_accessed = time.time()
        self._entries[prefix_hash] = entry

        logger.debug(
            f"[HybridCacheBackend] Stored {prefix_hash}: "
            f"{len(entry.blocks)} layers, hybrid={self.is_hybrid_cache(entry.blocks)}"
        )

    def supports_batch(self) -> bool:
        """
        Hybrid cache supports batch operations.

        Returns:
            True - Can merge/extract (with RNN state handling)
        """
        return True

    def merge(self, entries: List[CacheEntry]) -> CacheEntry:
        """
        Merge multiple hybrid cache entries.

        Args:
            entries: List of single-sequence CacheEntry objects

        Returns:
            New batched CacheEntry with merged RNN snapshots
        """
        if not entries:
            raise ValueError("Cannot merge empty entries")

        # Merge each layer
        try:
            from mlx_lm.models.cache import KVCache, BatchKVCache
            from vllm_mlx.utils.mamba_cache import BatchMambaCache

            merged_blocks = []
            for layer_idx in range(len(entries[0].blocks)):
                layer_caches = [e.blocks[layer_idx] for e in entries]

                # Determine cache type and merge appropriately
                if isinstance(layer_caches[0], KVCache):
                    merged = layer_caches[0].merge(layer_caches)
                elif hasattr(layer_caches[0], "state"):
                    # ArraysCache/MambaCache
                    merged = BatchMambaCache.merge(layer_caches)
                else:
                    raise ValueError(
                        f"Unknown cache type at layer {layer_idx}: "
                        f"{type(layer_caches[0])}"
                    )
                merged_blocks.append(merged)

            # Merge RNN snapshots
            merged_snapshots = {}
            for entry in entries:
                if entry.rnn_snapshots:
                    for snap_id, snap_data in entry.rnn_snapshots.items():
                        # Create batched snapshot ID
                        batched_id = f"batched_{snap_id}"
                        if batched_id not in merged_snapshots:
                            merged_snapshots[batched_id] = {}
                        merged_snapshots[batched_id].update(snap_data)

            merged_entry = CacheEntry(
                entry_id=f"hybrid_merged_{len(entries)}_{time.time():.0f}",
                prefix_hash="merged",
                cache_type="hybrid",
                blocks=merged_blocks,
                rnn_snapshots=merged_snapshots,
                tokens=sum(e.tokens for e in entries),
            )

            return merged_entry

        except Exception as e:
            logger.error(f"[HybridCacheBackend] Merge failed: {e}")
            raise

    def extract(self, entry: CacheEntry, idx: int) -> CacheEntry:
        """
        Extract single entry from batched hybrid cache.

        Args:
            entry: Batched CacheEntry
            idx: Index to extract

        Returns:
            Single-sequence CacheEntry
        """
        extracted_blocks = []
        for block in entry.blocks:
            if hasattr(block, "extract"):
                extracted_blocks.append(block.extract(idx))
            else:
                extracted_blocks.append(block)

        # Extract RNN snapshots (approximate - snapshots are per-batch)
        extracted_snapshots = {}
        if entry.rnn_snapshots:
            for snap_id, snap_data in entry.rnn_snapshots.items():
                # Create single-sequence snapshot ID
                single_id = f"single_{idx}_{snap_id}"
                extracted_snapshots[single_id] = snap_data

        return CacheEntry(
            entry_id=f"hybrid_extracted_{idx}",
            prefix_hash=entry.prefix_hash,
            cache_type="hybrid",
            blocks=extracted_blocks,
            rnn_snapshots=extracted_snapshots,
            tokens=entry.tokens,
        )

    def remove(self, prefix_hash: str) -> bool:
        """
        Remove hybrid cache entry.

        Args:
            prefix_hash: Entry hash to remove

        Returns:
            True if removed
        """
        if prefix_hash in self._entries:
            entry = self._entries[prefix_hash]
            # Discard any associated snapshots
            if entry.rnn_snapshots:
                for snap_id in entry.rnn_snapshots:
                    self.discard_snapshot(snap_id)
            del self._entries[prefix_hash]
            return True
        return False

    def clear(self) -> None:
        """
        Clear all entries and snapshots.
        """
        self._entries.clear()
        self._rnn_snapshots.clear()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def stats(self) -> dict:
        """
        Get hybrid cache statistics.

        Returns:
            Dict with stats including snapshot count
        """
        total_memory = sum(e.size_bytes for e in self._entries.values())
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0.0

        return {
            "entries": len(self._entries),
            "max_entries": self.max_entries,
            "memory_bytes": total_memory,
            "memory_mb": total_memory / (1024 * 1024),
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "hit_rate": hit_rate,
            "active_snapshots": len(self._rnn_snapshots),
            "supports_batch": True,
        }

    def _maybe_evict(self) -> None:
        """
        Evict if capacity exceeded.
        """
        if len(self._entries) >= self.max_entries:
            self._evict_one()

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
            victim_hash = min(
                self._entries.keys(),
                key=lambda h: self._entries[h].last_accessed,
            )
        elif self.eviction_policy == "lfu":
            victim_hash = min(
                self._entries.keys(),
                key=lambda h: self._entries[h].access_count,
            )
        else:
            victim_hash = min(
                self._entries.keys(),
                key=lambda h: self._entries[h].created_at,
            )

        self.remove(victim_hash)
        self._evictions += 1