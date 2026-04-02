# SPDX-License-Identifier: Apache-2.0
"""
Scheduler package for vllm-mlx continuous batching.

This package provides the Scheduler class and related components
for managing request scheduling using mlx-lm's BatchGenerator.

Public API (backward compatible with original scheduler.py):
- Scheduler: Main scheduler class
- SchedulerConfig: Configuration class
- SchedulerOutput: Output from scheduling steps
- SchedulingPolicy: Policy enum (FCFS, PRIORITY)

Architecture (Phase 3 refactor):
- config.py: Configuration classes
- policy.py: Scheduling policy enum
- chunked_prefill.py: Chunked prefill manager
- mtp.py: MTP speculative decoding
- core.py: Scheduler core logic
- utils.py: Utility functions
"""

# Import public API components
from .config import SchedulerConfig, SchedulerOutput
from .policy import SchedulingPolicy
from .core import Scheduler

# Public API exports
__all__ = [
    "Scheduler",
    "SchedulerConfig",
    "SchedulerOutput",
    "SchedulingPolicy",
]