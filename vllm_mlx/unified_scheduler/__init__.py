# SPDX-License-Identifier: Apache-2.0
"""
Unified Scheduler Module for vllm-mlx.

Provides a single scheduler that handles both LLM and MLLM requests:
- UnifiedScheduler: Request type-aware scheduling
- TextSchedulerBackend: mlx-lm BatchGenerator backend
- MultimodalSchedulerBackend: MLLMBatchGenerator backend

This module provides Phase 3 Engine abstraction layer.
"""

from .unified import (
    UnifiedScheduler,
    UnifiedSchedulerConfig,
    UnifiedRequest,
    RequestType,
    SchedulerBackend,
    TextSchedulerBackend,
    MultimodalSchedulerBackend,
)

__all__ = [
    "UnifiedScheduler",
    "UnifiedSchedulerConfig",
    "UnifiedRequest",
    "RequestType",
    "SchedulerBackend",
    "TextSchedulerBackend",
    "MultimodalSchedulerBackend",
]