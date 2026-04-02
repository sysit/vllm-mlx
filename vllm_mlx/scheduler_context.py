# SPDX-License-Identifier: Apache-2.0
"""
SchedulerContext and BatchGeneratorAdapter for eliminating circular dependencies.

This module provides:
- SchedulerContext: Shared state container for scheduler and batch generator
- BatchGeneratorAdapter: Clean wrapper around BatchGenerator that eliminates monkey-patching

The design follows the dependency injection pattern:
- SchedulerContext holds all shared state (requests, uid mappings, abort flags)
- BatchGeneratorAdapter accesses state through context, not direct references
- No monkey-patching of BatchGenerator internals
"""

import logging
import time as _time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import mlx.core as mx
from mlx_lm.generate import BatchGenerator

logger = logging.getLogger(__name__)


class SchedulerContext:
    """
    Shared state container for scheduler and batch generator.
    
    This eliminates circular dependencies by providing a single source of truth
    for all state that needs to be shared between Scheduler and BatchGenerator.
    
    Attributes:
        requests: All requests by ID (from Scheduler)
        uid_to_request_id: Mapping from BatchGenerator UID to request ID
        request_id_to_uid: Reverse mapping from request ID to UID
        pending_abort_ids: Set of request IDs queued for abort (thread-safe)
        running: Currently running requests by ID
        waiting: Queue of waiting requests
    """
    
    def __init__(self):
        """Initialize empty context."""
        # Request tracking (owned by Scheduler, shared with Adapter)
        self.requests: Dict[str, Any] = {}
        self.uid_to_request_id: Dict[int, str] = {}
        self.request_id_to_uid: Dict[str, int] = {}
        
        # Abort queue (thread-safe via GIL)
        self.pending_abort_ids: Set[str] = set()
        
        # Running/waiting tracking
        self.running: Dict[str, Any] = {}
        self.waiting: List[Any] = []  # deque proxy
        
        # Callbacks for cache management
        self.mid_prefill_save: Optional[Callable] = None
        self.prompt_cache_save: Optional[Callable] = None
        
    def get_request_by_uid(self, uid: int) -> Optional[Any]:
        """Get request by BatchGenerator UID."""
        request_id = self.uid_to_request_id.get(uid)
        if request_id is None:
            return None
        return self.requests.get(request_id)
    
    def get_request_id_by_uid(self, uid: int) -> Optional[str]:
        """Get request ID by BatchGenerator UID."""
        return self.uid_to_request_id.get(uid)
    
    def register_mapping(self, request_id: str, uid: int) -> None:
        """Register bidirectional mapping between request ID and UID."""
        self.uid_to_request_id[uid] = request_id
        self.request_id_to_uid[request_id] = uid
    
    def unregister_mapping(self, request_id: str) -> None:
        """Remove mapping for a finished/aborted request."""
        uid = self.request_id_to_uid.pop(request_id, None)
        if uid is not None:
            self.uid_to_request_id.pop(uid, None)
    
    def has_pending_aborts(self) -> bool:
        """Check if there are pending abort requests."""
        return bool(self.pending_abort_ids)
    
    def get_pending_aborts(self) -> Set[str]:
        """Get copy of pending abort IDs."""
        return self.pending_abort_ids.copy()
    
    def add_pending_abort(self, request_id: str) -> None:
        """Add request ID to pending abort queue."""
        self.pending_abort_ids.add(request_id)
    
    def clear_pending_abort(self, request_id: str) -> None:
        """Remove request ID from pending abort queue."""
        self.pending_abort_ids.discard(request_id)
    
    def clear(self) -> None:
        """Clear all state (for reset)."""
        self.requests.clear()
        self.uid_to_request_id.clear()
        self.request_id_to_uid.clear()
        self.pending_abort_ids.clear()
        self.running.clear()
        self.waiting.clear()


class BatchGeneratorAdapter:
    """
    Clean wrapper around mlx-lm BatchGenerator.
    
    This adapter provides a clean API for chunked prefill without monkey-patching
    BatchGenerator internals. It accesses shared state through SchedulerContext.
    
    Key differences from monkey-patching approach:
    - No modification of BatchGenerator._step, _next, or other internals
    - State access through context object (dependency injection)
    - Easier to test and maintain
    - Clear separation of concerns
    """
    
    def __init__(
        self,
        batch_gen: BatchGenerator,
        context: SchedulerContext,
        chunked_budget: int = 0,
        mid_prefill_save: Optional[Callable] = None,
        prompt_cache_save: Optional[Callable] = None,
    ):
        """
        Initialize adapter with BatchGenerator and shared context.
        
        Args:
            batch_gen: The underlying mlx-lm BatchGenerator
            context: Shared state container
            chunked_budget: Max tokens per prefill chunk (0 = disabled)
            mid_prefill_save: Optional callback for mid-prefill cache save
            prompt_cache_save: Optional callback for prompt-only cache save
        """
        self._batch_gen = batch_gen
        self._context = context
        self._chunked_budget = chunked_budget
        
        # Store callbacks in context for shared access
        context.mid_prefill_save = mid_prefill_save
        context.prompt_cache_save = prompt_cache_save
        
        # Partial prefill state (None when no prefill in progress)
        self._partial: Optional[Dict[str, Any]] = None
        
        # Install hooks via composition, not monkey-patching
        self._setup_hooks()
        
        if chunked_budget > 0:
            logger.info(f"[BatchGeneratorAdapter] chunked prefill enabled, budget={chunked_budget}")
    
    def _setup_hooks(self) -> None:
        """
        Setup hooks for chunked prefill using composition.
        
        Instead of monkey-patching BatchGenerator internals, we wrap its
        public methods and delegate to the underlying implementation.
        """
        # We don't modify _batch_gen internals at all
        # All chunked prefill logic is handled in our step() wrapper
        pass
    
    def insert(
        self,
        prompts: List[List[int]],
        max_tokens: List[int],
        caches: Optional[List[Any]] = None,
    ) -> List[int]:
        """
        Insert prompts into batch generator.
        
        Delegates directly to underlying BatchGenerator.
        """
        return self._batch_gen.insert(prompts, max_tokens=max_tokens, caches=caches)
    
    def remove(self, uids: List[int]) -> None:
        """
        Remove requests from batch generator.
        
        Clears partial state if any removed UID is being prefilled.
        """
        if self._partial is not None:
            partial_uids = set(self._partial["uids"])
            removed_uids = set(uids)
            if partial_uids & removed_uids:
                logger.info(
                    f"[BatchGeneratorAdapter] clearing partial state for removed UIDs: "
                    f"{partial_uids & removed_uids}"
                )
                self._partial = None
                mx.clear_cache()
        
        self._batch_gen.remove(uids)
    
    def next(self) -> List[Any]:
        """
        Run one generation step with chunked prefill support.
        
        This is the main entry point that replaces BatchGenerator._next()
        with chunked prefill logic, WITHOUT modifying the underlying object.
        """
        # If chunked prefill is disabled, delegate directly
        if self._chunked_budget <= 0:
            return self._batch_gen._next()
        
        # Otherwise, run our chunked logic
        return self._step_chunked()
    
    def _step_chunked(self) -> List[Any]:
        """
        Chunked prefill step logic.
        
        This implements the same logic as the monkey-patched version,
        but as a clean method that accesses state through context.
        """
        # Import helpers from mlx_lm (same as original monkey-patch)
        from mlx_lm.generate import (
            Batch,
            _left_pad_prompts,
            _make_cache,
            _merge_caches,
            _right_pad_prompts,
        )
        
        batch_gen = self._batch_gen
        context = self._context
        budget = self._chunked_budget
        
        # ----- Continue a partial prefill -----
        if self._partial is not None:
            # Check for pending aborts BEFORE processing next chunk
            partial_rids = {
                context.uid_to_request_id.get(u) 
                for u in self._partial["uids"]
            }
            aborted_rids = partial_rids & context.pending_abort_ids
            if aborted_rids:
                logger.info(
                    f"[chunked_prefill] abort detected mid-prefill, "
                    f"clearing partial for: {aborted_rids}"
                )
                self._partial = None
                mx.clear_cache()
                return self._generation_step()
            
            tic = _time.perf_counter()
            partial = self._partial
            inputs = partial["inputs"]
            prompt_cache = partial["cache"]
            remaining = inputs.shape[1]
            
            n_to_process = min(budget, remaining - 1) if remaining > 1 else 0
            
            if n_to_process > 0:
                batch_gen.model(
                    mx.contiguous(inputs[:, :n_to_process]),
                    cache=prompt_cache
                )
                mx.eval([c.state for c in prompt_cache])
                inputs = inputs[:, n_to_process:]
                partial["inputs"] = inputs
                partial["processed"] += n_to_process
                
                # Progress callback
                if batch_gen.prompt_progress_callback:
                    batch_gen.prompt_progress_callback([
                        (uid, partial["processed"], partial["total"])
                        for uid in partial["uids"]
                    ])
                
                # Save intermediate cache
                if context.mid_prefill_save is not None and len(partial["uids"]) == 1:
                    context.mid_prefill_save(
                        partial["uids"][0],
                        partial["processed"],
                        prompt_cache
                    )
                
                if partial.get("is_cached"):
                    mx.clear_cache()
            
            # Check if prefill is done
            if inputs.shape[1] <= 1:
                # Finalize
                if partial.get("is_cached"):
                    mx.eval([c.state for c in prompt_cache])
                    inputs = partial["last_inputs"]
                
                for c in prompt_cache:
                    c.finalize()
                mx.clear_cache()
                
                y, logprobs = batch_gen._step(
                    inputs,
                    prompt_cache,
                    partial["samplers"],
                    partial["logits_processors"],
                    partial["tokens"],
                )
                mx.async_eval(y, logprobs)
                
                new_batch = Batch(
                    list(partial["uids"]),
                    y,
                    logprobs,
                    list(partial["max_tokens"]),
                    [0] * len(partial["uids"]),
                    prompt_cache,
                    list(partial["samplers"]),
                    list(partial["logits_processors"]),
                    partial["tokens"],
                )
                
                # Save prompt-only cache
                if context.prompt_cache_save is not None and len(partial["uids"]) == 1:
                    uid = partial["uids"][0]
                    try:
                        context.prompt_cache_save(uid, new_batch.extract_cache(0))
                    except Exception:
                        pass
                
                if batch_gen.active_batch is None:
                    batch_gen.active_batch = new_batch
                else:
                    batch_gen.active_batch.extend(new_batch)
                
                self._partial = None
                batch_gen._stats.prompt_time += _time.perf_counter() - tic
            else:
                batch_gen._stats.prompt_time += _time.perf_counter() - tic
            
            return self._generation_step()
        
        # ----- No partial — check if next prompt needs chunking -----
        num_active = len(batch_gen.active_batch) if batch_gen.active_batch else 0
        num_to_add = batch_gen.completion_batch_size - num_active
        
        if num_to_add >= batch_gen.prefill_batch_size and batch_gen.unprocessed_prompts:
            batch_prompts = batch_gen.unprocessed_prompts[:batch_gen.prefill_batch_size]
            if batch_prompts:
                total_tokens = sum(len(p[1]) for p in batch_prompts)
                
                # Check for prefix boundary split
                needs_boundary_split = False
                for uid, toks, *_ in batch_prompts:
                    rid = context.uid_to_request_id.get(uid)
                    req = context.requests.get(rid) if rid else None
                    if req and getattr(req, "prefix_boundary", 0) > 0:
                        needs_boundary_split = True
                        break
                
                if total_tokens > budget or needs_boundary_split:
                    # Large prompt — start chunked prefill
                    return self._start_chunked_prefill(batch_prompts, needs_boundary_split)
                else:
                    # Small prompt — process directly
                    tic = _time.perf_counter()
                    
                    if batch_gen.active_batch is not None:
                        mx.eval(batch_gen.active_batch.y, batch_gen.active_batch.logprobs)
                        batch_gen._stats.generation_time += _time.perf_counter() - tic
                        tic = _time.perf_counter()
                    else:
                        mx.clear_cache()
                    
                    new_batch = batch_gen._process_prompts(batch_prompts)
                    batch_gen.unprocessed_prompts = batch_gen.unprocessed_prompts[
                        batch_gen.prefill_batch_size:
                    ]
                    
                    if batch_gen.active_batch is None:
                        batch_gen.active_batch = new_batch
                    else:
                        batch_gen.active_batch.extend(new_batch)
                    
                    batch_gen._stats.prompt_time += _time.perf_counter() - tic
                    return self._generation_step()
        
        # Pure generation or no work
        return self._generation_step()
    
    def _start_chunked_prefill(
        self,
        batch_prompts: List[Tuple],
        needs_boundary_split: bool,
    ) -> List[Any]:
        """Start a chunked prefill for large prompts."""
        from mlx_lm.generate import (
            _left_pad_prompts,
            _make_cache,
            _merge_caches,
            _right_pad_prompts,
        )
        
        batch_gen = self._batch_gen
        context = self._context
        budget = self._chunked_budget
        
        tic = _time.perf_counter()
        
        # Eval outstanding generation tokens
        if batch_gen.active_batch is not None:
            mx.eval(batch_gen.active_batch.y, batch_gen.active_batch.logprobs)
            batch_gen._stats.generation_time += _time.perf_counter() - tic
            tic = _time.perf_counter()
        else:
            mx.clear_cache()
        
        (
            uids,
            inputs_raw,
            max_tokens_list,
            caches,
            samplers,
            logits_processors,
            _prompt_checkpoints,
        ) = zip(*batch_prompts)
        lengths = [len(p) for p in inputs_raw]
        max_length = max(lengths)
        padding = [max_length - ln for ln in lengths]
        tokens = [mx.array(inp) for inp in inputs_raw]
        is_cached = not all(c[0].empty() for c in caches)
        
        batch_gen._stats.prompt_tokens += sum(lengths)
        
        if not is_cached:
            padded = _left_pad_prompts(inputs_raw, max_length=max_length)
            prompt_cache = _make_cache(batch_gen.model, padding, batch_gen.max_kv_size)
        else:
            last_inputs = mx.array([p[-1:] for p in inputs_raw])
            padded = _right_pad_prompts(inputs_raw, max_length=max_length)
            prompt_cache = _merge_caches(caches)
            for c in prompt_cache:
                c.prepare(
                    lengths=[ln - 1 for ln in lengths],
                    right_padding=padding,
                )
        
        # Remove from unprocessed
        batch_gen.unprocessed_prompts = batch_gen.unprocessed_prompts[
            batch_gen.prefill_batch_size:
        ]
        
        # Determine first chunk size
        first_chunk = budget
        if needs_boundary_split and len(batch_prompts) == 1:
            uid0 = uids[0]
            rid0 = context.uid_to_request_id.get(uid0)
            req0 = context.requests.get(rid0) if rid0 else None
            pb = getattr(req0, "prefix_boundary", 0) if req0 else 0
            cached = getattr(req0, "cached_tokens", 0) if req0 else 0
            adjusted_pb = pb - cached
            if 0 < adjusted_pb < padded.shape[1]:
                first_chunk = adjusted_pb
        
        n_to_process = min(first_chunk, padded.shape[1] - 1)
        if n_to_process > 0:
            batch_gen.model(
                mx.contiguous(padded[:, :n_to_process]),
                cache=prompt_cache,
            )
            mx.eval([c.state for c in prompt_cache])
            padded = padded[:, n_to_process:]
            if is_cached:
                mx.clear_cache()
        
        self._partial = {
            "uids": list(uids),
            "inputs": padded,
            "cache": prompt_cache,
            "tokens": tokens,
            "max_tokens": list(max_tokens_list),
            "samplers": list(samplers),
            "logits_processors": list(logits_processors),
            "processed": n_to_process,
            "total": max_length,
            "is_cached": is_cached,
        }
        if is_cached:
            self._partial["last_inputs"] = last_inputs
        
        # Progress callback
        if batch_gen.prompt_progress_callback:
            batch_gen.prompt_progress_callback([
                (uid, n_to_process, max_length)
                for uid in self._partial["uids"]
            ])
        
        # Save intermediate cache
        if context.mid_prefill_save is not None and len(uids) == 1:
            context.mid_prefill_save(uids[0], n_to_process, prompt_cache)
        
        batch_gen._stats.prompt_time += _time.perf_counter() - tic
        
        return self._generation_step()
    
    def _generation_step(self) -> List[Any]:
        """Run one generation step on active batch."""
        batch_gen = self._batch_gen
        batch = batch_gen.active_batch
        
        if batch is None or len(batch) == 0:
            return []
        
        tic_gen = _time.perf_counter()
        y, logprobs = batch.y, batch.logprobs
        
        for i, toks in enumerate(batch.tokens):
            batch.tokens[i] = mx.concatenate((toks, y[i:i+1]))
        
        batch.y, batch.logprobs = batch_gen._step(
            y[:, None],
            batch.cache,
            batch.samplers,
            batch.logits_processors,
            batch.tokens,
        )
        mx.async_eval(batch.y, batch.logprobs)
        
        y = y.tolist()
        batch_gen._stats.generation_time += _time.perf_counter() - tic_gen
        
        keep_idx = []
        end_idx = []
        responses = []
        
        for e, (t, uid, num_tok, max_tok) in enumerate(
            zip(y, batch.uids, batch.num_tokens, batch.max_tokens)
        ):
            cache_out = None
            num_tok += 1
            batch.num_tokens[e] = num_tok
            
            if t in batch_gen.stop_tokens:
                finish_reason = "stop"
                end_idx.append(e)
            elif num_tok >= max_tok:
                finish_reason = "length"
                end_idx.append(e)
            else:
                finish_reason = None
                keep_idx.append(e)
            
            if finish_reason is not None:
                cache_out = batch.extract_cache(e)
            
            responses.append(
                batch_gen.Response(uid, t, logprobs[e], finish_reason, cache_out)
            )
        
        if end_idx:
            if keep_idx:
                batch.filter(keep_idx)
            else:
                batch_gen.active_batch = None
        
        batch_gen._stats.generation_tokens += len(responses)
        return responses
    
    def close(self) -> None:
        """Close underlying BatchGenerator."""
        if hasattr(self._batch_gen, "close"):
            self._batch_gen.close()
    
    # Expose underlying BatchGenerator properties for compatibility
    @property
    def active_batch(self) -> Optional[Any]:
        """Get active batch from underlying generator."""
        return self._batch_gen.active_batch
    
    @property
    def unprocessed_prompts(self) -> List[Any]:
        """Get unprocessed prompts queue."""
        return self._batch_gen.unprocessed_prompts
    
    @property
    def stop_tokens(self) -> Set[int]:
        """Get stop tokens set."""
        return self._batch_gen.stop_tokens
    
    @property
    def model(self) -> Any:
        """Get model."""
        return self._batch_gen.model
    
    @property
    def sampler(self) -> Any:
        """Get sampler."""
        return self._batch_gen.sampler
    
    @property
    def prefill_batch_size(self) -> int:
        """Get prefill batch size."""
        return self._batch_gen.prefill_batch_size
    
    @property
    def completion_batch_size(self) -> int:
        """Get completion batch size."""
        return self._batch_gen.completion_batch_size
    
    @property
    def max_kv_size(self) -> Optional[int]:
        """Get max KV size."""
        return self._batch_gen.max_kv_size
    
    @property
    def Response(self) -> type:
        """Get Response class."""
        return self._batch_gen.Response
    
    @property
    def _stats(self) -> Any:
        """Get stats object."""
        return self._batch_gen._stats
    
    @property
    def prompt_progress_callback(self) -> Optional[Callable]:
        """Get prompt progress callback."""
        return self._batch_gen.prompt_progress_callback
    
    def _process_prompts(self, prompts: List[Tuple]) -> Any:
        """Delegate to underlying _process_prompts."""
        return self._batch_gen._process_prompts(prompts)