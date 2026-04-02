#!/usr/bin/env python3
"""
Phase 2 refactor validation script.

Tests SchedulerContext and BatchGeneratorAdapter without requiring real model.
"""

import sys
from unittest.mock import MagicMock

# Test 1: SchedulerContext functionality
print("=" * 60)
print("Test 1: SchedulerContext")
print("=" * 60)

from vllm_mlx.scheduler_context import SchedulerContext

ctx = SchedulerContext()

# Test mapping
ctx.register_mapping("req-123", 42)
assert ctx.get_request_id_by_uid(42) == "req-123"
assert ctx.get_request_id_by_uid(999) is None
print("✓ Mapping registration works")

# Test pending aborts
ctx.add_pending_abort("req-123")
assert ctx.has_pending_aborts()
assert "req-123" in ctx.get_pending_aborts()
ctx.clear_pending_abort("req-123")
assert not ctx.has_pending_aborts()
print("✓ Pending abort management works")

# Test request tracking
mock_request = MagicMock()
mock_request.request_id = "req-123"
ctx.requests["req-123"] = mock_request
assert ctx.get_request_by_uid(42) == mock_request
print("✓ Request tracking works")

# Test clear
ctx.clear()
assert len(ctx.requests) == 0
assert len(ctx.uid_to_request_id) == 0
print("✓ Context clear works")

print("\n" + "=" * 60)
print("Test 2: Scheduler with SchedulerContext")
print("=" * 60)

from vllm_mlx.scheduler import Scheduler, SchedulerConfig
from vllm_mlx.request import Request, SamplingParams

model = MagicMock()
tokenizer = MagicMock()
tokenizer.encode = lambda x: list(range(len(x.split())))
tokenizer.decode = lambda x: " ".join(str(t) for t in x)
tokenizer.eos_token_id = 0
tokenizer.eos_token_ids = {0}

config = SchedulerConfig(max_num_seqs=10)
scheduler = Scheduler(model, tokenizer, config)

# Verify context is initialized
assert scheduler._context is not None
print("✓ Scheduler has SchedulerContext")

# Verify context shares state with scheduler
assert scheduler._context.requests == scheduler.requests
assert scheduler._context.uid_to_request_id == scheduler.uid_to_request_id
print("✓ Context state is shared with scheduler")

# Test abort request (uses context)
request = Request(
    request_id="test-1",
    prompt="Hello world",
    sampling_params=SamplingParams(max_tokens=10),
)
scheduler.add_request(request)

result = scheduler.abort_request("test-1")
assert result is True
assert scheduler._context.has_pending_aborts()
print("✓ Abort request uses context")

# Process aborts
scheduler._process_pending_aborts()
assert not scheduler._context.has_pending_aborts()
print("✓ Process aborts clears context")

# Test reset clears context
scheduler.reset()
assert len(scheduler._context.requests) == 0
assert not scheduler._context.has_pending_aborts()
print("✓ Reset clears context")

print("\n" + "=" * 60)
print("Test 3: BatchGeneratorAdapter (mock test)")
print("=" * 60)

from vllm_mlx.scheduler_context import BatchGeneratorAdapter
from mlx_lm.generate import BatchGenerator

# Create mock BatchGenerator
mock_bg = MagicMock(spec=BatchGenerator)
mock_bg.active_batch = None
mock_bg.unprocessed_prompts = []
mock_bg.prefill_batch_size = 8
mock_bg.completion_batch_size = 32
mock_bg.stop_tokens = set([0])
mock_bg.model = model
mock_bg.insert = MagicMock(return_value=[1])
mock_bg.remove = MagicMock()
mock_bg.close = MagicMock()

# Create adapter
adapter = BatchGeneratorAdapter(
    batch_gen=mock_bg,
    context=ctx,
    chunked_budget=512,
)

# Test insert delegates
uids = adapter.insert([[1, 2, 3]], max_tokens=[10])
assert uids == [1]
print("✓ Adapter.insert delegates to BatchGenerator")

# Test remove delegates
adapter.remove([1])
mock_bg.remove.assert_called_once_with([1])
print("✓ Adapter.remove delegates to BatchGenerator")

# Test properties
assert adapter.prefill_batch_size == 8
assert adapter.completion_batch_size == 32
print("✓ Adapter properties work")

# Test close
adapter.close()
mock_bg.close.assert_called_once()
print("✓ Adapter.close works")

print("\n" + "=" * 60)
print("ALL TESTS PASSED!")
print("=" * 60)
print("\nPhase 2 refactor validation successful:")
print("- SchedulerContext eliminates circular dependencies")
print("- BatchGeneratorAdapter replaces monkey-patching")
print("- Scheduler state sharing works correctly")
print("- Abort handling uses context")
print("\nReady for integration testing.")