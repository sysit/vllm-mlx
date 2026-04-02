# SPDX-License-Identifier: Apache-2.0
"""
Scheduling policy definitions.

Extracted from scheduler.py as part of Phase 3 architecture refactor.
"""

from enum import Enum


class SchedulingPolicy(Enum):
    """Scheduling policy for request ordering."""

    FCFS = "fcfs"  # First-Come-First-Served
    PRIORITY = "priority"  # Priority-based