# SPDX-License-Identifier: Apache-2.0
"""
Unified cache configuration for vllm-mlx.

This module provides a unified configuration class for all cache types,
including memory-aware, paged, prefix, and vision caches.
"""

from dataclasses import dataclass
from typing import Optional

# Constants
_BYTES_PER_MB = 1024 * 1024
_DEFAULT_MEMORY_PERCENT = 0.20  # 20% of available RAM
_MIN_MEMORY_BYTES = 100 * _BYTES_PER_MB  # Minimum 100MB
_DEFAULT_BLOCK_SIZE = 64  # tokens per block
_DEFAULT_MAX_BLOCKS = 1000
_DEFAULT_MAX_ENTRIES = 1000  # Safety limit for cache entries


@dataclass(frozen=True)
class CacheConfig:
    """
    Unified configuration for all cache types in vllm-mlx.

    This class consolidates configuration options for:
    - Memory-aware prefix cache
    - Paged block cache
    - Vision embedding cache
    - MLLM prefix cache

    Attributes:
        max_memory_mb: Maximum memory in MB for caches. If None, auto-detects.
        max_memory_percent: Fraction of available RAM to use (0.0-1.0).
        max_entries: Maximum number of cache entries.
        block_size: Number of tokens per block for paged cache.
        max_blocks: Maximum number of blocks for paged cache.
        enable_caching: Whether to enable prefix caching.
        enable_memory_tracking: Whether to track per-entry memory.
        kv_quantize: Whether to quantize KV cache layers.
        kv_bits: Number of bits for KV cache quantization.
        kv_group_size: Group size for KV cache quantization.
        kv_min_quantize_tokens: Minimum sequence length for quantization.
        max_pixel_entries: Maximum entries for vision pixel cache.
        max_encoding_entries: Maximum entries for vision encoding cache.
    """

    # Memory limits
    max_memory_mb: Optional[int] = None
    max_memory_percent: float = _DEFAULT_MEMORY_PERCENT

    # Entry limits
    max_entries: int = _DEFAULT_MAX_ENTRIES
    max_blocks: int = _DEFAULT_MAX_BLOCKS

    # Block settings
    block_size: int = _DEFAULT_BLOCK_SIZE
    enable_caching: bool = True

    # Memory tracking
    enable_memory_tracking: bool = True

    # KV quantization
    kv_quantize: bool = False
    kv_bits: int = 8
    kv_group_size: int = 64
    kv_min_quantize_tokens: int = 256

    # Vision cache
    max_pixel_entries: int = 50
    max_encoding_entries: int = 20

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if not 0.0 < self.max_memory_percent <= 1.0:
            raise ValueError(
                f"max_memory_percent must be in (0, 1], got {self.max_memory_percent}"
            )
        if self.max_entries < 1:
            raise ValueError(f"max_entries must be >= 1, got {self.max_entries}")
        if self.max_blocks < 1:
            raise ValueError(f"max_blocks must be >= 1, got {self.max_blocks}")
        if self.block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {self.block_size}")
        if self.kv_min_quantize_tokens < 0:
            raise ValueError(
                f"kv_min_quantize_tokens must be >= 0, got {self.kv_min_quantize_tokens}"
            )
        if self.max_pixel_entries < 0:
            raise ValueError(
                f"max_pixel_entries must be >= 0, got {self.max_pixel_entries}"
            )
        if self.max_encoding_entries < 0:
            raise ValueError(
                f"max_encoding_entries must be >= 0, got {self.max_encoding_entries}"
            )

    def compute_memory_limit(self) -> int:
        """
        Compute the memory limit in bytes.

        Returns:
            Memory limit in bytes.
        """
        if self.max_memory_mb is not None:
            return self.max_memory_mb * _BYTES_PER_MB

        # Auto-detect from available system memory
        try:
            import psutil

            available = psutil.virtual_memory().available
            limit = int(available * self.max_memory_percent)
            return max(limit, _MIN_MEMORY_BYTES)
        except ImportError:
            # Fallback if psutil not available
            return _MIN_MEMORY_BYTES

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "max_memory_mb": self.max_memory_mb,
            "max_memory_percent": self.max_memory_percent,
            "max_entries": self.max_entries,
            "max_blocks": self.max_blocks,
            "block_size": self.block_size,
            "enable_caching": self.enable_caching,
            "enable_memory_tracking": self.enable_memory_tracking,
            "kv_quantize": self.kv_quantize,
            "kv_bits": self.kv_bits,
            "kv_group_size": self.kv_group_size,
            "kv_min_quantize_tokens": self.kv_min_quantize_tokens,
            "max_pixel_entries": self.max_pixel_entries,
            "max_encoding_entries": self.max_encoding_entries,
        }


# Backward compatibility aliases
MemoryCacheConfig = CacheConfig