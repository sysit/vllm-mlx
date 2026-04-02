# SPDX-License-Identifier: Apache-2.0
"""
Chunked prefill manager for large prompt handling.

Extracted from scheduler.py as part of Phase 3 architecture refactor.

This module provides the _install_chunked_prefill function that monkey-patches
a BatchGenerator instance to break large prefills into chunks, preventing
starvation of active generation requests during long prefills.
"""

import logging
import time as _time
from typing import Any, Callable, Dict, List, Optional, Set

import mlx.core as mx

logger = logging.getLogger(__name__)


def install_chunked_prefill(
    batch_gen: Any,
    budget: int,
    mid_prefill_save: Optional[Callable] = None,
    prompt_cache_save: Optional[Callable] = None,
    pending_abort_ids: Optional[Set[str]] = None,
    uid_to_request_id: Optional[Dict[int, str]] = None,
    requests: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Monkey-patch a BatchGenerator instance so that large prefills are
    broken into chunks of at most *budget* tokens each.

    Between chunks the generation loop gets a chance to produce one token
    for every active request, preventing starvation during long prefills.

    Args:
        batch_gen: The BatchGenerator to patch.
        budget: Max tokens per prefill chunk.
        mid_prefill_save: Optional callback(uid, processed, prompt_cache)
            called after each chunk to save intermediate KV cache state.
        prompt_cache_save: Optional callback for prompt-only cache save
        pending_abort_ids: Set of request IDs queued for abort
        uid_to_request_id: Mapping from UID to request ID
        requests: All requests by ID
    """
    from mlx_lm.generate import (
        Batch,
        _left_pad_prompts,
        _make_cache,
        _merge_caches,
        _right_pad_prompts,
    )

    # Keep references to originals
    _orig_next = batch_gen._next
    _orig_remove = batch_gen.remove
    _orig_process_prompts = batch_gen._process_prompts

    # Partial prefill state (None when no prefill in progress)
    batch_gen._partial = None

    # Monkey-patch _process_prompts to capture prompt-only cache state.
    # At the point where _process_prompts returns, the Batch cache contains
    # the exact prompt-only state: all prompt tokens have been processed
    # through the model, but no output token has been fed back yet.
    # This is the only safe capture point for hybrid Mamba+Transformer
    # models whose MambaCache state is cumulative.
    if prompt_cache_save is not None:

        def _patched_process_prompts(prompts, _self=batch_gen):
            batch = _orig_process_prompts(prompts)
            for e, uid in enumerate(batch.uids):
                if batch.num_tokens[e] == 0:
                    try:
                        prompt_cache_save(uid, batch.extract_cache(e))
                    except Exception:
                        pass
            return batch

        batch_gen._process_prompts = _patched_process_prompts

    def _generation_step(self=batch_gen):
        """Run one generation step on the active batch. Returns responses."""
        batch = self.active_batch
        if batch is None or len(batch) == 0:
            return []

        tic_gen = _time.perf_counter()
        y, logprobs = batch.y, batch.logprobs
        for i, toks in enumerate(batch.tokens):
            batch.tokens[i] = mx.concatenate((toks, y[i : i + 1]))
        batch.y, batch.logprobs = self._step(
            y[:, None],
            batch.cache,
            batch.samplers,
            batch.logits_processors,
            batch.tokens,
        )
        mx.async_eval(batch.y, batch.logprobs)

        y = y.tolist()
        self._stats.generation_time += _time.perf_counter() - tic_gen

        keep_idx = []
        end_idx = []
        responses = []
        for e, (t, uid, num_tok, max_tok) in enumerate(
            zip(y, batch.uids, batch.num_tokens, batch.max_tokens)
        ):
            cache_out = None
            num_tok += 1
            batch.num_tokens[e] = num_tok
            if t in self.stop_tokens:
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
                self.Response(uid, t, logprobs[e], finish_reason, cache_out)
            )

        if len(end_idx):
            if len(keep_idx) > 0:
                batch.filter(keep_idx)
            else:
                self.active_batch = None

        self._stats.generation_tokens += len(responses)
        return responses

    def _chunked_next(self=batch_gen):  # noqa: C901
        """
        Replacement for _next() that chunks large prefills.

        Only intercepts when:
        1. A partial prefill is in progress (_partial is not None)
        2. The next prompt batch exceeds the budget

        Everything else delegates to the original _next().
        """
        # ----- Continue a partial prefill -----
        if self._partial is not None:
            # Check for pending aborts BEFORE processing next chunk
            if pending_abort_ids is not None and uid_to_request_id is not None:
                partial_rids = {uid_to_request_id.get(u) for u in self._partial["uids"]}
                aborted_rids = partial_rids & pending_abort_ids
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
                self.model(mx.contiguous(inputs[:, :n_to_process]), cache=prompt_cache)
                mx.eval([c.state for c in prompt_cache])
                inputs = inputs[:, n_to_process:]
                partial["inputs"] = inputs
                partial["processed"] += n_to_process

                self.prompt_progress_callback(
                    [
                        (uid, partial["processed"], partial["total"])
                        for uid in partial["uids"]
                    ]
                )

                # Save intermediate cache for disconnect resilience
                if mid_prefill_save is not None and len(partial["uids"]) == 1:
                    mid_prefill_save(
                        partial["uids"][0], partial["processed"], prompt_cache
                    )

                if partial.get("is_cached"):
                    mx.clear_cache()

            # Check if prefill is done (only 1 token left or 0)
            if inputs.shape[1] <= 1:
                # Finalize
                if partial.get("is_cached"):
                    mx.eval([c.state for c in prompt_cache])
                    inputs = partial["last_inputs"]

                for c in prompt_cache:
                    c.finalize()
                mx.clear_cache()

                y, logprobs = self._step(
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

                # Save prompt-only cache BEFORE merging into active batch.
                # This is the chunked-prefill equivalent of the
                # _patched_process_prompts hook — at this point the cache
                # contains the exact prompt-only state (num_tokens == 0).
                if prompt_cache_save is not None and len(partial["uids"]) == 1:
                    uid = partial["uids"][0]
                    try:
                        prompt_cache_save(uid, new_batch.extract_cache(0))
                    except Exception:
                        pass

                if self.active_batch is None:
                    self.active_batch = new_batch
                else:
                    self.active_batch.extend(new_batch)

                self._partial = None
                self._stats.prompt_time += _time.perf_counter() - tic
            else:
                # Not done yet — record prompt time for this chunk
                self._stats.prompt_time += _time.perf_counter() - tic

            # Generation step for active requests between chunks
            return self._generation_step()

        # ----- No partial — check if next prompt batch needs chunking -----
        num_active = len(self.active_batch) if self.active_batch else 0
        num_to_add = self.completion_batch_size - num_active

        if num_to_add >= self.prefill_batch_size and self.unprocessed_prompts:
            batch_prompts = self.unprocessed_prompts[: self.prefill_batch_size]
            if batch_prompts:
                total_tokens = sum(len(p[1]) for p in batch_prompts)

                # Check if any prompt has a prefix_boundary that
                # requires two-phase prefill for cache save at that boundary.
                _needs_boundary_split = False
                if requests is not None and uid_to_request_id is not None:
                    for _uid, _toks, *_ in batch_prompts:
                        _rid = uid_to_request_id.get(_uid)
                        _req = requests.get(_rid) if _rid else None
                        if _req and getattr(_req, "prefix_boundary", 0) > 0:
                            _needs_boundary_split = True
                            break

                if total_tokens > budget or _needs_boundary_split:
                    # Large prompt batch or prefix boundary — start partial prefill
                    tic = _time.perf_counter()

                    # Eval outstanding generation tokens before switching.
                    # Also drain pending async_eval when active_batch is None
                    # (previous request finished) — stale async_eval work on
                    # generation_stream can block subsequent model forwards.
                    if self.active_batch is not None:
                        mx.eval(self.active_batch.y, self.active_batch.logprobs)
                        self._stats.generation_time += _time.perf_counter() - tic
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

                    self._stats.prompt_tokens += sum(lengths)

                    if not is_cached:
                        padded = _left_pad_prompts(inputs_raw, max_length=max_length)
                        prompt_cache = _make_cache(
                            self.model, padding, self.max_kv_size
                        )
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
                    self.unprocessed_prompts = self.unprocessed_prompts[
                        self.prefill_batch_size :
                    ]

                    # Process first chunk — if prefix_boundary is set,
                    # use it as the first chunk size so that mid_prefill_save
                    # can capture the exact prefix cache state (critical for
                    # hybrid Mamba+Transformer models where trim is unsafe).
                    # When the request already has cached tokens (cache hit),
                    # adjust the boundary relative to the remaining tokens.
                    _first_chunk = budget
                    if _needs_boundary_split and len(batch_prompts) == 1:
                        _uid0 = uids[0]
                        _rid0 = uid_to_request_id.get(_uid0)
                        _req0 = requests.get(_rid0) if _rid0 else None
                        _pb = getattr(_req0, "prefix_boundary", 0) if _req0 else 0
                        _cached = getattr(_req0, "cached_tokens", 0) if _req0 else 0
                        _adjusted_pb = _pb - _cached
                        if 0 < _adjusted_pb < padded.shape[1]:
                            _first_chunk = _adjusted_pb
                    n_to_process = min(_first_chunk, padded.shape[1] - 1)
                    if n_to_process > 0:
                        self.model(
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

                    self.prompt_progress_callback(
                        [
                            (uid, n_to_process, max_length)
                            for uid in self._partial["uids"]
                        ]
                    )

                    # Save intermediate cache for disconnect resilience
                    if mid_prefill_save is not None and len(uids) == 1:
                        mid_prefill_save(uids[0], n_to_process, prompt_cache)

                    self._stats.prompt_time += _time.perf_counter() - tic

                    # Generation step for active requests
                    return self._generation_step()

                else:
                    # Small prompt batch — process directly without _orig_next.
                    # _orig_next's while loop processes multiple batches per call
                    # which causes batch-dimension mismatches in DeltaRNN conv_state
                    # when mixing prefix-cached and fresh prompts.
                    # Processing one batch per _next call avoids this.
                    tic = _time.perf_counter()

                    # Eval outstanding generation tokens before prefill.
                    # Also drain when active_batch is None to clear stale
                    # async_eval work from the previous request.
                    if self.active_batch is not None:
                        mx.eval(self.active_batch.y, self.active_batch.logprobs)
                        self._stats.generation_time += _time.perf_counter() - tic
                        tic = _time.perf_counter()
                    else:
                        mx.clear_cache()

                    new_batch = self._process_prompts(batch_prompts)
                    self.unprocessed_prompts = self.unprocessed_prompts[
                        self.prefill_batch_size :
                    ]

                    if self.active_batch is None:
                        self.active_batch = new_batch
                    else:
                        self.active_batch.extend(new_batch)

                    self._stats.prompt_time += _time.perf_counter() - tic
                    return self._generation_step()

        # Pure generation or no work — run generation step directly
        return self._generation_step()

    def _patched_remove(uids_to_remove, _self=batch_gen):
        """Clear partial state if aborted request is being prefilled."""
        if _self._partial is not None:
            partial_uids = set(_self._partial["uids"])
            if partial_uids & set(uids_to_remove):
                logger.info(
                    f"[chunked_prefill] clearing partial state for aborted uids: "
                    f"{partial_uids & set(uids_to_remove)}"
                )
                _self._partial = None
                mx.clear_cache()  # flush Metal encoders after dropping partial state
        _orig_remove(uids_to_remove)

    batch_gen._next = _chunked_next
    batch_gen._generation_step = _generation_step
    batch_gen.remove = _patched_remove

    logger.info(f"[chunked_prefill] installed with budget={budget} tokens per step")