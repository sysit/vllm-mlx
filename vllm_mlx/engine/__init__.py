# SPDX-License-Identifier: Apache-2.0
"""
Engine abstraction for vllm-mlx inference.

Provides two engine implementations:
- SimpleEngine: Direct model calls for maximum single-user throughput
- BatchedEngine: Continuous batching for multiple concurrent users

Also provides EngineRegistry for dynamic engine selection:
- EngineRegistry: Request type-aware routing
- select_engine(): Convenience function for engine selection
- create_engine(): Convenience function for engine instantiation

Also re-exports core engine components for backwards compatibility.
"""

from .base import BaseEngine, GenerationOutput
from .simple import SimpleEngine
from .batched import BatchedEngine
from .registry import (
    EngineRegistry,
    get_registry,
    select_engine,
    create_engine,
)

# Re-export from parent engine.py for backwards compatibility
from ..engine_core import EngineCore, AsyncEngineCore, EngineConfig

__all__ = [
    "BaseEngine",
    "GenerationOutput",
    "SimpleEngine",
    "BatchedEngine",
    # Engine Registry
    "EngineRegistry",
    "get_registry",
    "select_engine",
    "create_engine",
    # Core engine components
    "EngineCore",
    "AsyncEngineCore",
    "EngineConfig",
]
