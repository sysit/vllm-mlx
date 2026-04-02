# SPDX-License-Identifier: Apache-2.0
"""
Scheduler utility functions and constants.

Extracted from scheduler.py as part of Phase 3 architecture refactor.
"""

# Error patterns that indicate cache corruption
CACHE_CORRUPTION_PATTERNS = [
    "'NoneType' object is not subscriptable",
    "cache",
    "BatchKVCache",
]


def is_cache_corruption_error(error: Exception) -> bool:
    """
    Check if an error indicates cache corruption.

    Args:
        error: The exception to check

    Returns:
        True if the error matches cache corruption patterns
    """
    error_str = str(error)
    return any(pattern in error_str for pattern in CACHE_CORRUPTION_PATTERNS)