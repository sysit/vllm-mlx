# SPDX-License-Identifier: Apache-2.0
"""
vllm-mlx: Apple Silicon MLX backend for vLLM

This package provides native Apple Silicon GPU acceleration for vLLM
using Apple's MLX framework, mlx-lm for LLMs, and mlx-vlm for
vision-language models.

Features:
- Continuous batching via vLLM-style scheduler
- OpenAI-compatible API server
- Support for LLM and multimodal models

New unified cache module (v0.3.0+):
    from vllm_mlx.cache import (
        PrefixCache, BlockCache, PagedCache,
        VisionCache, MLLMCache, MemoryAwarePrefixCache
    )

Legacy imports still work for backward compatibility:
    from vllm_mlx import PrefixCacheManager, BlockAwarePrefixCache
"""

__version__ = "0.2.5"

# All imports are lazy to allow usage on non-Apple Silicon platforms
# (e.g., CI running on Linux) where mlx_lm is not available.


def __getattr__(name):
    """Lazy load all components to avoid mlx_lm import on non-Apple platforms."""
    # Request management
    if name in ("Request", "RequestOutput", "RequestStatus", "SamplingParams"):
        from vllm_mlx import request

        return getattr(request, name)

    # Scheduler
    if name in ("Scheduler", "SchedulerConfig", "SchedulerOutput"):
        from vllm_mlx import scheduler

        return getattr(scheduler, name)

    # Engine
    if name in ("EngineCore", "AsyncEngineCore", "EngineConfig"):
        from vllm_mlx import engine_core

        return getattr(engine_core, name)

    # New unified cache module exports (v0.3.0+)
    # All cache classes now live in vllm_mlx.cache module
    if name in (
        "PrefixCache",
        "BlockCache",
        "PagedCache",
        "VisionCache",
        "MLLMCache",
        "CacheConfig",
        # Legacy names (backward compatibility)
        "PrefixCacheManager",
        "PrefixCacheStats",
        "BlockAwarePrefixCache",
        "PagedCacheManager",
        "CacheBlock",
        "BlockTable",
        "CacheStats",
        "MLLMCacheManager",
        "MLLMCacheStats",
        "VLMCacheManager",
        "VLMCacheStats",
        "MemoryAwarePrefixCache",
        "MemoryCacheConfig",
        "VisionEmbeddingCache",
    ):
        from vllm_mlx import cache

        # Map legacy VLM names to MLLM
        if name.startswith("VLM"):
            name = name.replace("VLM", "MLLM")
        return getattr(cache, name)

    # Model registry
    if name in ("get_registry", "ModelOwnershipError"):
        from vllm_mlx import model_registry

        return getattr(model_registry, name)

    # vLLM integration components (require torch)
    if name == "MLXPlatform":
        from vllm_mlx.vllm_platform import MLXPlatform

        return MLXPlatform
    if name == "MLXWorker":
        from vllm_mlx.worker import MLXWorker

        return MLXWorker
    if name == "MLXModelRunner":
        from vllm_mlx.model_runner import MLXModelRunner

        return MLXModelRunner
    if name == "MLXAttentionBackend":
        from vllm_mlx.attention import MLXAttentionBackend

        return MLXAttentionBackend

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Core (lazy loaded, require torch)
    "MLXPlatform",
    "MLXWorker",
    "MLXModelRunner",
    "MLXAttentionBackend",
    # Request management
    "Request",
    "RequestOutput",
    "RequestStatus",
    "SamplingParams",
    # Scheduler
    "Scheduler",
    "SchedulerConfig",
    "SchedulerOutput",
    # Engine
    "EngineCore",
    "AsyncEngineCore",
    "EngineConfig",
    # Model registry
    "get_registry",
    "ModelOwnershipError",
    # New unified cache exports (v0.3.0+)
    "PrefixCache",
    "BlockCache",
    "PagedCache",
    "VisionCache",
    "MLLMCache",
    "CacheConfig",
    # Prefix cache (LLM) - legacy names
    "PrefixCacheManager",
    "PrefixCacheStats",
    "BlockAwarePrefixCache",
    # Paged cache (memory efficiency) - legacy names
    "PagedCacheManager",
    "CacheBlock",
    "BlockTable",
    "CacheStats",
    # MLLM cache (images/videos)
    "MLLMCacheManager",
    "MLLMCacheStats",
    # Legacy aliases
    "VLMCacheManager",
    "VLMCacheStats",
    # Version
    "__version__",
]
