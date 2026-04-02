# SPDX-License-Identifier: Apache-2.0
"""
Scheduler for vllm-mlx continuous batching.

COMPATIBILITY LAYER (Phase 3 architecture refactor)

This file now serves as a backward-compatible import facade.
All actual implementation has been moved to the scheduler/ package:

- scheduler/config.py: SchedulerConfig, SchedulerOutput
- scheduler/policy.py: SchedulingPolicy
- scheduler/core.py: Scheduler core logic
- scheduler/chunked_prefill.py: Chunked prefill manager
- scheduler/mtp.py: MTP speculative decoding
- scheduler/utils.py: Utility functions

Imports from this module will continue to work unchanged:
    from vllm_mlx.scheduler import Scheduler, SchedulerConfig
    from vllm_mlx.scheduler import SchedulingPolicy, SchedulerOutput

See scheduler/ package for detailed implementation.
"""

# Re-export all public API from the new package structure
from vllm_mlx.scheduler import (
    Scheduler,
    SchedulerConfig,
    SchedulerOutput,
    SchedulingPolicy,
)

# Also export internal functions for backward compatibility
# (these were used by tests and other internal modules)
from vllm_mlx.scheduler.chunked_prefill import install_chunked_prefill as _install_chunked_prefill
from vllm_mlx.scheduler.mtp import install_mtp as _install_mtp
from vllm_mlx.scheduler.utils import CACHE_CORRUPTION_PATTERNS

# Maintain backward compatibility for all exports
__all__ = [
    # Public API
    "Scheduler",
    "SchedulerConfig",
    "SchedulerOutput",
    "SchedulingPolicy",
    # Internal (for backward compatibility)
    "_install_chunked_prefill",
    "_install_mtp",
    "CACHE_CORRUPTION_PATTERNS",
]