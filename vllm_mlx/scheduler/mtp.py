# SPDX-License-Identifier: Apache-2.0
"""
MTP (Multi-Token Prediction) speculative decoding.

Extracted from scheduler.py as part of Phase 3 architecture refactor.

NOTE: This module uses monkey-patching as a LEGACY DESIGN pattern.
This approach is kept for simplicity and backward compatibility.

Future architecture improvements should integrate MTP support directly
into BatchGeneratorAdapter instead of monkey-patching BatchGenerator.
This would provide:
  - Better encapsulation and testability
  - Cleaner integration with chunked prefill
  - Explicit type contracts instead of dynamic patching

The monkey-patch pattern works but has known drawbacks:
  - Harder to debug (method replacement at runtime)
  - State shared via closures (_skip_state, _deferred_drafts)
  - Potential conflicts with other patches

For now, this legacy design is acceptable because:
  - MTP is an optional performance optimization
  - The patch is isolated to a single install_mtp() function
  - It works reliably with both KVCache and hybrid MambaCache models

See scheduler/context.py:BatchGeneratorAdapter for the preferred
adapter-based approach used for chunked prefill.
"""

import logging
from typing import Any, Dict, List, Optional

import mlx.core as mx
from mlx_lm.sample_utils import make_sampler

logger = logging.getLogger(__name__)


def install_mtp(
    batch_gen: Any,
    model: Any,
    num_draft_tokens: int = 1,
    optimistic: bool = False,
) -> None:
    """
    Monkey-patch a BatchGenerator to use MTP (Multi-Token Prediction)
    with always-advance strategy for hybrid MambaCache + KVCache.

    Flow per generation step:
    1. Use skip_state logits/hidden OR run model forward -> sample primary
    2. MTP head drafts one token after primary
    3. Verify [primary, draft] in one model call (always advances cache)
    4. Accept: skip_state from pos 1, defer draft for next step emission
       Reject: trim KVCache by 1, skip_state from pos 0 (no cold start)
    5. Draft is emitted in the NEXT generation step after primary

    Args:
        batch_gen: The BatchGenerator to patch
        model: The MLX model with MTP head
        num_draft_tokens: Number of draft tokens from MTP head
        optimistic: Skip acceptance check for max speed
    """
    _orig_step = batch_gen._step

    # Greedy sampler for MTP draft tokens
    _draft_sampler = make_sampler(temp=0.0)

    # Skip state: when MTP accepts, the cache already consumed [primary, draft].
    # Next _step call receives primary as input but must NOT re-feed it.
    # Instead, use stored logits from the verify pass.
    # Format: {'logits': (B, V), 'hidden': (B, 1, H)}
    _skip_state = [None]

    # Deferred drafts: draft tokens to emit in the NEXT generation step,
    # keyed by UID for stability across batch changes.
    # Format: {uid: {'token': int, 'logprobs': mx.array}}
    _deferred_drafts: Dict[int, Dict[str, Any]] = {}

    # MTP stats
    _mtp_stats = {"accepted": 0, "rejected": 0, "errors": 0}

    def _mtp_step(
        input_tokens,
        prompt_cache,
        samplers,
        logits_processors,
        tokens,
    ):
        """
        Extended _step with MTP always-advance strategy.

        Every step (after skip):
        1. Use skip_state logits/hidden OR run model forward
        2. Sample primary token P
        3. MTP head drafts token D
        4. Verify [P, D] in one model call (always advances cache)
        5. Accept: skip_state from position 1 (after D), defer D
           Reject: trim KVCache by 1, skip_state from position 0 (after P)

        No snapshot/restore — eliminates cold starts after rejection.
        MambaCache layers accept minor pollution on reject (exponential decay).

        During prefill (multi-token input), MTP is skipped entirely.
        """
        batch_size = input_tokens.shape[0]

        # --- Prefill guard: skip MTP for multi-token input,
        # during _process_prompts (active_batch not yet set), or when
        # the cache doesn't belong to the active batch (e.g. during
        # _process_prompts in the 2nd+ iteration of _orig_next's loop
        # or during _chunked_next partial prefill finalization).
        if (
            input_tokens.shape[1] > 1
            or batch_gen.active_batch is None
            or prompt_cache is not batch_gen.active_batch.cache
        ):
            _skip_state[0] = None
            return _orig_step(
                input_tokens,
                prompt_cache,
                samplers,
                logits_processors,
                tokens,
            )

        # --- Check skip state from previous MTP step ---
        skip = _skip_state[0]
        if skip is not None:
            if skip["logits"].shape[0] != batch_size:
                # Batch size changed since skip was stored — invalidate
                skip = None
                _skip_state[0] = None

        if skip is not None:
            # Skip mode: model already processed input_tokens during
            # previous verify. Use stored logits + hidden instead.
            logits = skip["logits"]
            hidden_states = skip["hidden"]
            _skip_state[0] = None
        else:
            # Normal model forward
            model_output = model(input_tokens, cache=prompt_cache, return_hidden=True)
            if isinstance(model_output, tuple):
                logits, hidden_states = model_output
            else:
                # Model doesn't support return_hidden — fall back
                return _orig_step(
                    input_tokens,
                    prompt_cache,
                    samplers,
                    logits_processors,
                    tokens,
                )
            logits = logits[:, -1, :]

        # --- Apply logits processors + sample primary ---
        if any(logits_processors):
            processed_logits = []
            for e in range(batch_size):
                sample_logits = logits[e : e + 1]
                for processor in logits_processors[e]:
                    sample_logits = processor(tokens[e], sample_logits)
                processed_logits.append(sample_logits)
            logits = mx.concatenate(processed_logits, axis=0)

        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        if any(samplers):
            all_samples = []
            for e in range(batch_size):
                sample_sampler = samplers[e] or batch_gen.sampler
                sampled = sample_sampler(logprobs[e : e + 1])
                all_samples.append(sampled)
            primary_tokens = mx.concatenate(all_samples, axis=0)
        else:
            primary_tokens = batch_gen.sampler(logprobs)

        # Get current UIDs (guaranteed non-empty: prefill guard above
        # prevents MTP from running when active_batch is None).
        current_uids = list(batch_gen.active_batch.uids)

        # --- MTP draft + always-advance verify ---
        try:
            # Draft: predict token n+2 from hidden states + primary (n+1)
            draft_logits = model.mtp_forward(
                hidden_states[:, -1:, :],
                primary_tokens[:, None],
                mtp_cache=None,
            )
            draft_logits = draft_logits[:, -1, :]
            draft_logprobs = draft_logits - mx.logsumexp(
                draft_logits, axis=-1, keepdims=True
            )
            draft_tokens = _draft_sampler(draft_logprobs)

            # Always-advance: feed [primary, draft] and let cache advance.
            #
            # Hybrid models (e.g. Qwen3-Next) mix attention (KVCache) and
            # recurrent layers (MambaCache/DeltaRNN).  KVCache supports
            # trim(1) to undo the draft token on reject, but recurrent
            # state is irreversible — rejected drafts permanently pollute
            # the RNN state, causing progressive output corruption.
            #
            # For hybrid models we snapshot recurrent state before verify
            # and on reject: trim KV by 2 (remove both P and D), restore
            # RNN snapshot, then re-advance with just P so both cache
            # types end up consistent at [..., P].
            _rnn_snapshots: Dict[int, List] = {}
            for _ci, _c in enumerate(prompt_cache):
                if not (hasattr(_c, "is_trimmable") and _c.is_trimmable()):
                    if hasattr(_c, "state"):
                        _rnn_snapshots[_ci] = [
                            s.copy() if s is not None else None for s in _c.state
                        ]

            verify_input = mx.concatenate(
                [primary_tokens[:, None], draft_tokens[:, None]], axis=1
            )
            verify_output = model(verify_input, cache=prompt_cache, return_hidden=True)
            if isinstance(verify_output, tuple):
                verify_logits, verify_hidden = verify_output
            else:
                verify_logits = verify_output
                verify_hidden = None

            if optimistic:
                # --- OPTIMISTIC: always accept, zero sync ---
                if verify_hidden is not None:
                    _skip_state[0] = {
                        "logits": verify_logits[:, 1, :],
                        "hidden": verify_hidden[:, -1:, :],
                    }
                    verify_lp = verify_logits[:, 0, :] - mx.logsumexp(
                        verify_logits[:, 0, :], axis=-1, keepdims=True
                    )
                    mx.async_eval(
                        _skip_state[0]["logits"],
                        _skip_state[0]["hidden"],
                        draft_tokens,
                        verify_lp,
                    )
                    for e in range(batch_size):
                        uid = current_uids[e]
                        _deferred_drafts[uid] = {
                            "token_array": draft_tokens[e : e + 1],
                            "logprobs": verify_lp[e],
                        }
                else:
                    _skip_state[0] = None
                _mtp_stats["accepted"] += 1
            else:
                # --- VERIFIED MODE: single eval + Python comparison ---
                verify_pred = mx.argmax(verify_logits[:, 0, :], axis=-1)
                mx.eval(verify_pred, draft_tokens)
                pred_list = verify_pred.tolist()
                draft_list = draft_tokens.tolist()
                all_accepted = pred_list == draft_list

                if all_accepted and verify_hidden is not None:
                    # --- ACCEPT ---
                    _skip_state[0] = {
                        "logits": verify_logits[:, 1, :],
                        "hidden": verify_hidden[:, -1:, :],
                    }
                    mx.async_eval(_skip_state[0]["logits"], _skip_state[0]["hidden"])
                    verify_lp = verify_logits[:, 0, :] - mx.logsumexp(
                        verify_logits[:, 0, :], axis=-1, keepdims=True
                    )
                    for e in range(batch_size):
                        uid = current_uids[e]
                        _deferred_drafts[uid] = {
                            "token": draft_list[e],
                            "logprobs": verify_lp[e],
                        }
                    _mtp_stats["accepted"] += 1

                else:
                    # --- REJECT (always-advance) ---
                    if _rnn_snapshots:
                        # Hybrid model: undo the entire verify pass
                        # (both P and D) for all cache types, then
                        # re-advance with just P for a consistent state.
                        for c in prompt_cache:
                            if (
                                hasattr(c, "is_trimmable")
                                and c.is_trimmable()
                                and hasattr(c, "trim")
                            ):
                                c.trim(2)
                        for _ci, _snap in _rnn_snapshots.items():
                            prompt_cache[_ci].state = _snap
                        # Re-advance with primary only — both KV and RNN
                        # now advance by exactly 1 (the primary token).
                        rerun_out = model(
                            primary_tokens[:, None],
                            cache=prompt_cache,
                            return_hidden=True,
                        )
                        if isinstance(rerun_out, tuple):
                            rerun_logits, rerun_hidden = rerun_out
                        else:
                            rerun_logits = rerun_out
                            rerun_hidden = None
                        if rerun_hidden is not None:
                            _skip_state[0] = {
                                "logits": rerun_logits[:, -1, :],
                                "hidden": rerun_hidden[:, -1:, :],
                            }
                            mx.async_eval(
                                _skip_state[0]["logits"],
                                _skip_state[0]["hidden"],
                            )
                        else:
                            _skip_state[0] = None
                    else:
                        # Pure attention model: simple trim(1) is enough.
                        for c in prompt_cache:
                            if (
                                hasattr(c, "is_trimmable")
                                and c.is_trimmable()
                                and hasattr(c, "trim")
                            ):
                                c.trim(1)
                        if verify_hidden is not None:
                            _skip_state[0] = {
                                "logits": verify_logits[:, 0, :],
                                "hidden": verify_hidden[:, 0:1, :],
                            }
                            mx.async_eval(
                                _skip_state[0]["logits"],
                                _skip_state[0]["hidden"],
                            )
                        else:
                            _skip_state[0] = None
                    for uid in current_uids:
                        _deferred_drafts.pop(uid, None)
                    _mtp_stats["rejected"] += 1

        except Exception as e:
            logger.debug(f"[MTP] draft/verify failed: {e}")
            _skip_state[0] = None
            _mtp_stats["errors"] += 1

        return primary_tokens, list(logprobs)

    # Wrap _next() to emit deferred MTP drafts after each primary token.
    # This works regardless of whether _chunked_next or original _next is
    # the current _next implementation, because it sits at the top level.
    # Store as attribute so it's always the correct reference, even after
    # BatchGenerator recreation.
    batch_gen._inner_next = batch_gen._next

    def _mtp_next(self=batch_gen):
        """Wrapper around _next that emits deferred MTP draft tokens.

        After each primary token, if the previous step's MTP draft was
        accepted, it is emitted as an additional response.
        """
        # Clear stale MTP state when no batch is active.
        # This prevents skip_state/deferred_drafts from a finished request
        # from leaking into the next request and causing stale computation
        # graph references on generation_stream.
        if self.active_batch is None:
            _skip_state[0] = None
            _deferred_drafts.clear()

        # Save deferred drafts from PREVIOUS step before _inner_next
        # runs _mtp_step, which may store NEW deferred drafts.
        prev_deferred: Dict[int, Dict[str, Any]] = {}
        if self.active_batch is not None:
            for uid in self.active_batch.uids:
                if uid in _deferred_drafts:
                    prev_deferred[uid] = _deferred_drafts.pop(uid)

        # Run the inner _next (original or chunked) — calls _mtp_step
        responses = self._inner_next()

        if not prev_deferred or not responses:
            return responses

        # Augment responses with deferred drafts from the previous step.
        # The Response from _next reports the OLD batch.y (the primary
        # from the *previous* _step call). The deferred draft follows
        # that primary in the token stream, so emit it AFTER the primary.
        augmented = []
        draft_end_uids = set()
        for r in responses:
            uid = r.uid

            # Emit the primary response first
            augmented.append(r)

            if r.finish_reason is not None:
                # Sequence ended with primary — discard any pending draft
                _deferred_drafts.pop(uid, None)
                prev_deferred.pop(uid, None)
                continue

            # Emit deferred draft AFTER its primary
            if uid in prev_deferred:
                draft_info = prev_deferred.pop(uid)
                if "token" in draft_info:
                    draft_t = draft_info["token"]
                else:
                    draft_t = draft_info["token_array"].item()
                draft_lp = draft_info["logprobs"]

                if draft_t in self.stop_tokens:
                    augmented.append(
                        self.Response(uid, draft_t, draft_lp, "stop", None)
                    )
                    draft_end_uids.add(uid)
                else:
                    draft_finish = None
                    batch = self.active_batch
                    if batch is not None:
                        for e, bu in enumerate(batch.uids):
                            if bu == uid:
                                batch.num_tokens[e] += 1
                                batch.tokens[e] = mx.concatenate(
                                    (batch.tokens[e], mx.array([draft_t]))
                                )
                                if batch.num_tokens[e] >= batch.max_tokens[e]:
                                    draft_finish = "length"
                                    draft_end_uids.add(uid)
                                break

                    draft_cache_out = None
                    if draft_finish is not None and batch is not None:
                        for e, bu in enumerate(batch.uids):
                            if bu == uid:
                                draft_cache_out = batch.extract_cache(e)
                                break

                    augmented.append(
                        self.Response(
                            uid, draft_t, draft_lp, draft_finish, draft_cache_out
                        )
                    )

        # Remove sequences that finished due to draft tokens
        if draft_end_uids and self.active_batch is not None:
            keep = [
                e
                for e, u in enumerate(self.active_batch.uids)
                if u not in draft_end_uids
            ]
            if keep:
                self.active_batch.filter(keep)
            else:
                self.active_batch = None

        return augmented

    batch_gen._step = _mtp_step
    batch_gen._next = _mtp_next

    mode_str = "optimistic (no verify)" if optimistic else "always-advance"
    logger.info(
        f"[MTP] installed with num_draft_tokens={num_draft_tokens}, " f"{mode_str} mode"
    )