# SPDX-License-Identifier: Apache-2.0
"""
VisionCacheBackend - Vision embedding cache for MLLM models.

Vision embeddings are computed from image/video inputs and can be
expensive to recompute. This backend provides caching for vision
encoder outputs.

Key limitations:
- NO batch support: Each vision embedding is computed for a specific
  request context (pixel values, image grid, etc.) and cannot be merged
- Content hash based: Hash computed from image content, not tokens
- Request-scoped: Vision embeddings are tied to specific requests
"""

import hashlib
import logging
import time
from typing import Any, Optional, List, Dict

from .backend import CacheBackend, CacheEntry

logger = logging.getLogger(__name__)


class VisionCacheBackend(CacheBackend):
    """
    Vision embedding cache backend for MLLM models.

    Vision embeddings differ from KV cache in important ways:
    1. Cannot be batched - each embedding is request-specific
    2. Hashed by image content, not text tokens
    3. Includes pixel_values, image_grid_thw, and encoder outputs

    This backend stores vision encoder outputs keyed by image content hash,
    allowing reuse when the same image appears in different requests.
    However, vision embeddings cannot be merged across requests due to
    architectural constraints in current MLLM implementations.
    """

    def __init__(
        self,
        max_entries: int = 50,
        max_memory_mb: Optional[int] = None,
        eviction_policy: str = "lru",
    ):
        """
        Initialize vision cache backend.

        Args:
            max_entries: Maximum cache entries (smaller than KV due to size)
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

    def _compute_image_hash(
        self,
        image_data: Any,
        pixel_values: Optional[Any] = None,
    ) -> str:
        """
        Compute hash from image content.

        Args:
            image_data: Image path, URL, or base64 data
            pixel_values: Optional pixel values array

        Returns:
            Hash string
        """
        # If we have pixel_values array, hash that directly
        if pixel_values is not None:
            if hasattr(pixel_values, "tobytes"):
                arr_bytes = pixel_values.tobytes()
            else:
                arr_bytes = str(pixel_values).encode()
            return hashlib.sha256(arr_bytes).hexdigest()[:16]

        # Otherwise hash the image path/URL/data
        if isinstance(image_data, str):
            return hashlib.sha256(image_data.encode()).hexdigest()[:16]
        elif isinstance(image_data, bytes):
            return hashlib.sha256(image_data).hexdigest()[:16]
        else:
            return hashlib.sha256(str(image_data).encode()).hexdigest()[:16]

    def lookup(self, prefix_hash: str) -> Optional[CacheEntry]:
        """
        Look up vision embedding by content hash.

        Args:
            prefix_hash: Hash of image content

        Returns:
            CacheEntry if found, None otherwise
        """
        entry = self._entries.get(prefix_hash)
        if entry is None:
            self._misses += 1
            return None

        # Validate vision embedding
        if entry.vision_embeddings is None:
            logger.debug(f"[VisionCacheBackend] Entry {prefix_hash} has no embeddings")
            self.remove(prefix_hash)
            self._misses += 1
            return None

        # Update access stats
        entry.update_access(time.time())
        self._hits += 1
        return entry

    def store(self, prefix_hash: str, entry: CacheEntry) -> None:
        """
        Store vision embedding entry.

        Args:
            prefix_hash: Hash of image content
            entry: Cache entry with vision_embeddings
        """
        # Validate entry has vision embeddings
        if entry.vision_embeddings is None:
            logger.warning(
                "[VisionCacheBackend] Cannot store entry without vision_embeddings"
            )
            return

        # Check capacity
        self._maybe_evict()

        # Store entry
        entry.created_at = time.time()
        entry.last_accessed = time.time()
        entry.cache_type = "vision"
        self._entries[prefix_hash] = entry

        logger.debug(
            f"[VisionCacheBackend] Stored entry {prefix_hash}, "
            f"size={entry.size_bytes / 1024:.1f}KB"
        )

    def supports_batch(self) -> bool:
        """
        Vision cache does NOT support batch operations.

        Returns:
            False - Vision embeddings cannot be merged
        """
        return False

    def merge(self, entries: List[CacheEntry]) -> CacheEntry:
        """
        Vision cache does NOT support merge.

        Raises:
            NotImplementedError: Always - vision embeddings cannot be batched
        """
        raise NotImplementedError(
            "VisionCacheBackend does not support batch operations. "
            "Vision embeddings are request-specific and cannot be merged. "
            "Each request must use its own vision cache."
        )

    def extract(self, entry: CacheEntry, idx: int) -> CacheEntry:
        """
        Vision cache does NOT support extract.

        Raises:
            NotImplementedError: Always
        """
        raise NotImplementedError(
            "VisionCacheBackend does not support batch operations."
        )

    def remove(self, prefix_hash: str) -> bool:
        """
        Remove vision cache entry.

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
        Clear all vision cache entries.
        """
        self._entries.clear()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def stats(self) -> dict:
        """
        Get vision cache statistics.

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
            "supports_batch": False,
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
        logger.debug(f"[VisionCacheBackend] Evicted {victim_hash}")

    def store_from_request(
        self,
        image_hash: str,
        pixel_values: Any,
        encoder_outputs: Any,
        image_grid_thw: Optional[Any] = None,
        request_id: str = "",
    ) -> CacheEntry:
        """
        Convenience method to store vision cache from request data.

        Args:
            image_hash: Pre-computed image content hash
            pixel_values: Pixel values array
            encoder_outputs: Vision encoder outputs
            image_grid_thw: Optional image grid info
            request_id: Request ID for logging

        Returns:
            Created CacheEntry
        """
        vision_embeddings = {
            "pixel_values": pixel_values,
            "encoder_outputs": encoder_outputs,
            "image_grid_thw": image_grid_thw,
        }

        entry = CacheEntry(
            entry_id=f"vision_{request_id}_{time.time():.0f}",
            prefix_hash=image_hash,
            cache_type="vision",
            blocks=[],  # No KV blocks for vision-only cache
            vision_embeddings=vision_embeddings,
        )

        self.store(image_hash, entry)
        return entry