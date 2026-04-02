# SPDX-License-Identifier: Apache-2.0
"""
Unified cache module for vllm-mlx.

This module provides a unified interface for all cache types:
- PrefixCache: Trie-based LRU prefix cache
- BlockCache: Block-based prefix cache with PagedCache integration
- MemoryAwarePrefixCache: Memory-aware prefix cache with automatic limits
- VisionCache: Vision embedding cache for MLLM
- MLLMCache: Multimodal prefix cache with image hashing
- PagedCache: Paged KV cache manager (vLLM BlockPool style)

All legacy class names are available as aliases:
- PrefixCacheManager → PrefixCache
- BlockAwarePrefixCache → BlockCache
- PagedCacheManager → PagedCache
- MLLMPrefixCacheManager → MLLMCache
- VisionEmbeddingCache → VisionCache

Usage:
    # New unified imports (recommended)
    from vllm_mlx.cache import PrefixCache, BlockCache, PagedCache

    # Legacy imports (backward compatibility)
    from vllm_mlx.cache import PrefixCacheManager, BlockAwarePrefixCache
"""

# Configuration
from .config import CacheConfig, MemoryCacheConfig

# Paged cache (base for BlockCache)
from .paged_cache import (
    BlockTable,
    BlockHash,
    BlockHashToBlockMap,
    FreeKVCacheBlockQueue,
    compute_block_hash,
    PagedCache,
    CacheBlock,
    CacheStats as PagedCacheStats,  # Block-level stats
    KVCacheBlock,  # Backward compatibility
)

# Prefix cache (trie-based LRU)
from .prefix_cache import (
    CacheEntry,
    PrefixCacheStats,
    PrefixCache,
)

# Block cache (block-aware prefix cache)
from .block_cache import (
    BlockCacheEntry,
    BlockCache,
)

# Memory-aware cache
from .memory_cache import (
    MemoryAwarePrefixCache,
    CacheStats,  # Memory-aware cache stats (hits, misses, etc.)
    estimate_kv_cache_memory,
    _CacheEntry,  # Export for testing
    _array_memory,  # Export for testing
    _get_available_memory,  # Export for testing
    _quantize_cache,  # Export for testing
    _dequantize_cache,  # Export for testing
    _trim_to_offset,  # Export for testing
)

# Vision cache
from .vision_cache import (
    VisionCacheStats,
    PixelCacheEntry,
    PixelOnlyCacheEntry,
    EncodingCacheEntry,
    VisionCache,
)

# MLLM cache
from .mllm_cache import (
    MLLMCacheStats,
    MLLMCacheEntry,
    MLLMCache,
    compute_image_hash,  # Export for testing
    compute_images_hash,  # Export for testing
)


# ---------------------------------------------------------------------
# Backward compatibility aliases
# ---------------------------------------------------------------------
# These aliases allow legacy imports to work seamlessly:
#   from vllm_mlx.cache import PrefixCacheManager  # works
#   from vllm_mlx import PrefixCacheManager        # works (via __init__.py)
# ---------------------------------------------------------------------

# Paged cache aliases
PagedCacheManager = PagedCache
KVCacheBlock = CacheBlock

# Prefix cache aliases
PrefixCacheManager = PrefixCache

# Block cache aliases
BlockAwarePrefixCache = BlockCache

# Vision cache aliases
VisionEmbeddingCache = VisionCache

# MLLM cache aliases
MLLMPrefixCacheManager = MLLMCache
MLLMCacheManager = MLLMCache
MLLMPrefixCacheEntry = MLLMCacheEntry  # Legacy naming for tests
VLMPrefixCacheManager = MLLMCache  # Legacy VLM naming
VLMCacheManager = MLLMCache        # Legacy VLM naming

# Memory cache aliases (for backward compatibility)
MemoryCacheStats = CacheStats


__all__ = [
    # Configuration
    "CacheConfig",
    "MemoryCacheConfig",
    # Paged cache
    "CacheBlock",
    "CacheStats",
    "PagedCacheStats",  # Block-level stats
    "BlockTable",
    "BlockHash",
    "BlockHashToBlockMap",
    "FreeKVCacheBlockQueue",
    "compute_block_hash",
    "PagedCache",
    "PagedCacheManager",  # Backward compatibility
    "KVCacheBlock",  # Backward compatibility
    # Prefix cache
    "CacheEntry",
    "PrefixCacheStats",
    "PrefixCache",
    "PrefixCacheManager",  # Backward compatibility
    # Block cache
    "BlockCacheEntry",
    "BlockCache",
    "BlockAwarePrefixCache",  # Backward compatibility
    # Memory cache
    "MemoryAwarePrefixCache",
    "MemoryCacheStats",  # Alias for CacheStats
    "estimate_kv_cache_memory",
    "_CacheEntry",  # For testing
    "_array_memory",  # For testing
    "_get_available_memory",  # For testing
    "_quantize_cache",  # For testing
    "_dequantize_cache",  # For testing
    "_trim_to_offset",  # For testing
    # Vision cache
    "VisionCacheStats",
    "PixelCacheEntry",
    "PixelOnlyCacheEntry",
    "EncodingCacheEntry",
    "VisionCache",
    "VisionEmbeddingCache",  # Backward compatibility
    # MLLM cache
    "MLLMCacheStats",
    "MLLMCacheEntry",
    "MLLMCache",
    "MLLMPrefixCacheManager",  # Backward compatibility
    "MLLMCacheManager",
    "MLLMPrefixCacheEntry",  # Legacy for tests
    "compute_image_hash",  # For testing
    "compute_images_hash",  # For testing
    "VLMPrefixCacheManager",  # Legacy alias
    "VLMCacheManager",  # Legacy alias
]