# SPDX-License-Identifier: Apache-2.0
"""
Unified Cache Backend system for vllm-mlx.

Provides three cache backend types:
1. KVCacheBackend - Standard LLM text cache (batch/merge supported)
2. VisionCacheBackend - Vision embedding cache (no batch support)
3. HybridCacheBackend - Mamba+Transformer hybrid model cache

Also provides Block-based cache management with CoW prefix sharing:
4. BlockManager - Block-based KV cache allocation (PagedAttention-style)
5. CoWPrefixSharing - Copy-on-Write prefix sharing for multi-turn efficiency
6. PrefixSharingIntegration - Integration layer with KVCacheBackend
"""

from .backend import CacheBackend, CacheEntry
from .kv_backend import KVCacheBackend
from .vision_backend import VisionCacheBackend
from .hybrid_backend import HybridCacheBackend
from .block_manager import BlockManager, KVBlock, BlockTable
from .cow import (
    CoWPrefixSharing,
    PrefixEntry,
    PrefixSharingIntegration,
)

__all__ = [
    "CacheBackend",
    "CacheEntry",
    "KVCacheBackend",
    "VisionCacheBackend",
    "HybridCacheBackend",
    # Block-based cache
    "BlockManager",
    "KVBlock",
    "BlockTable",
    # CoW prefix sharing
    "CoWPrefixSharing",
    "PrefixEntry",
    "PrefixSharingIntegration",
]