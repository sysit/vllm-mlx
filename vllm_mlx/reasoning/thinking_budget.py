# SPDX-License-Identifier: Apache-2.0
"""
Thinking budget logits processor for controlling thinking token generation.

This module provides ThinkingBudgetCriteria, a logits processor that
forces the model to end its thinking phase after a specified number of tokens.
It works by detecting thinking start/end markers and suppressing the end token
after the budget is exhausted.

Compatible with models like Qwen3, DeepSeek-R1 that use thinking tags.
"""

import logging
from typing import Callable

import mlx.core as mx

logger = logging.getLogger(__name__)


class ThinkingBudgetCriteria:
    """
    Logits processor that limits thinking token generation.

    This processor counts tokens generated within thinking blocks and
    forces the model to emit the thinking end token when the budget
    is exhausted.

    Usage:
        ```python
        from vllm_mlx.reasoning.thinking_budget import ThinkingBudgetCriteria

        # Create the criteria
        criteria = ThinkingBudgetCriteria(
            tokenizer=tokenizer,
            thinking_budget=1000,
            thinking_start_token="<think>",
            thinking_end_token="</think>",
        )

        # Pass to generate_step as logits_processors=[criteria]
        for token, logprobs in generate_step(..., logits_processors=[criteria]):
            ...
        ```

    How it works:
        1. Tracks whether we're inside a thinking block (after start token)
        2. Counts tokens generated while in thinking mode
        3. When budget is exceeded, boosts the end token logit to force emission
        4. After end token is emitted, stops interfering
    """

    def __init__(
        self,
        tokenizer,
        thinking_budget: int,
        thinking_start_token: str = "<think>",
        thinking_end_token: str = "</think>",
        enable_thinking: bool = True,
    ):
        """
        Initialize the thinking budget criteria.

        Args:
            tokenizer: The tokenizer for encoding/decoding tokens.
            thinking_budget: Maximum number of thinking tokens to generate.
            thinking_start_token: The marker that starts thinking (e.g., "<think>").
            thinking_end_token: The marker that ends thinking (e.g., "</think>").
            enable_thinking: Whether thinking mode is enabled.
        """
        self.tokenizer = tokenizer
        self.thinking_budget = thinking_budget
        self.thinking_start_token = thinking_start_token
        self.thinking_end_token = thinking_end_token
        self.enable_thinking = enable_thinking

        # Get token IDs for start/end markers
        # Note: Some tokenizers may encode these as multiple tokens
        self.start_token_ids = tokenizer.encode(thinking_start_token, add_special_tokens=False)
        self.end_token_ids = tokenizer.encode(thinking_end_token, add_special_tokens=False)

        # State tracking
        self.in_thinking = False
        self.thinking_token_count = 0
        self.end_emitted = False

        # Track partial token matching for multi-token markers
        self.start_match_pos = 0
        self.end_match_pos = 0
        self.recent_tokens = []  # Recent token IDs for marker detection

        logger.debug(
            f"ThinkingBudgetCriteria initialized: budget={thinking_budget}, "
            f"start_ids={self.start_token_ids}, end_ids={self.end_token_ids}"
        )

    def __call__(
        self,
        tokens: mx.array,
        logits: mx.array,
    ) -> mx.array:
        """
        Process logits to enforce thinking budget.

        Args:
            tokens: Array of generated token IDs so far.
            logits: Logits for the next token prediction.

        Returns:
            Modified logits (or original if no modification needed).
        """
        if not self.enable_thinking or self.end_emitted:
            # Thinking disabled or already ended - don't interfere
            return logits

        # Get the last generated token
        last_token = int(tokens[-1].item())

        # Update state based on token
        self._update_state(last_token)

        # Check if we need to force end token
        if self.in_thinking and self.thinking_token_count >= self.thinking_budget:
            # Budget exhausted - force the end token
            if self.end_token_ids:
                # Boost the first token of the end marker
                end_token_id = self.end_token_ids[0]
                # Set all logits to very low except the end token
                modified_logits = logits.copy()
                # Suppress all other tokens
                modified_logits[:] = -100.0  # Very low log prob
                # Boost end token
                modified_logits[end_token_id] = 100.0  # Very high log prob
                logger.debug(
                    f"Thinking budget exhausted ({self.thinking_token_count} tokens), "
                    f"forcing end token {end_token_id}"
                )
                return modified_logits

        return logits

    def _update_state(self, token: int) -> None:
        """
        Update internal state based on generated token.

        Tracks whether we're inside/outside thinking blocks by
        detecting start/end markers.

        Args:
            token: The last generated token ID.
        """
        self.recent_tokens.append(token)

        # Keep only recent tokens needed for marker detection
        max_marker_len = max(len(self.start_token_ids), len(self.end_token_ids))
        if len(self.recent_tokens) > max_marker_len + 5:
            self.recent_tokens = self.recent_tokens[-(max_marker_len + 5):]

        # Detect start marker (enter thinking)
        if not self.in_thinking:
            self.start_match_pos = self._check_marker_match(
                self.recent_tokens, self.start_token_ids, self.start_match_pos
            )
            if self.start_match_pos == len(self.start_token_ids):
                # Full start marker detected
                self.in_thinking = True
                self.thinking_token_count = 0
                self.start_match_pos = 0
                logger.debug("Entered thinking block")

        # Detect end marker (exit thinking)
        if self.in_thinking:
            self.end_match_pos = self._check_marker_match(
                self.recent_tokens, self.end_token_ids, self.end_match_pos
            )
            if self.end_match_pos == len(self.end_token_ids):
                # Full end marker detected
                self.in_thinking = False
                self.end_emitted = True
                self.end_match_pos = 0
                logger.debug(
                    f"Exited thinking block after {self.thinking_token_count} tokens"
                )

        # Count thinking tokens
        if self.in_thinking:
            # Don't count start/end marker tokens
            if self.start_match_pos == 0 and self.end_match_pos == 0:
                self.thinking_token_count += 1

    def _check_marker_match(
        self,
        recent_tokens: list[int],
        marker_ids: list[int],
        current_pos: int,
    ) -> int:
        """
        Check if recent tokens match a marker sequence.

        Args:
            recent_tokens: List of recently generated token IDs.
            marker_ids: Token IDs of the marker to detect.
            current_pos: Current match position (how many tokens matched so far).

        Returns:
            Updated match position (0 if no match, len(marker_ids) if full match).
        """
        if not marker_ids:
            return 0

        # Check if the last token continues the marker sequence
        if current_pos < len(marker_ids):
            expected_token = marker_ids[current_pos]
            if recent_tokens and recent_tokens[-1] == expected_token:
                return current_pos + 1
            elif recent_tokens and recent_tokens[-1] == marker_ids[0]:
                # Restart match from beginning
                return 1
            else:
                # No match
                return 0

        return current_pos

    def reset(self) -> None:
        """
        Reset internal state for a new generation.
        """
        self.in_thinking = False
        self.thinking_token_count = 0
        self.end_emitted = False
        self.start_match_pos = 0
        self.end_match_pos = 0
        self.recent_tokens = []


def create_thinking_budget_criteria(
    tokenizer,
    thinking_budget: int,
    model_type: str = "qwen3",
    enable_thinking: bool = True,
) -> ThinkingBudgetCriteria | None:
    """
    Factory function to create ThinkingBudgetCriteria for specific models.

    Args:
        tokenizer: The tokenizer for the model (required).
        thinking_budget: Maximum thinking tokens.
        model_type: Model type for marker selection ("qwen3", "deepseek_r1", etc).
        enable_thinking: Whether thinking is enabled.

    Returns:
        ThinkingBudgetCriteria instance, or None if thinking_budget <= 0,
        enable_thinking=False, or tokenizer is None.
    """
    if thinking_budget <= 0 or not enable_thinking or tokenizer is None:
        return None

    # Model-specific markers
    markers = {
        "qwen3": ("<think>", "</think>"),
        "qwen3.5": ("<think>", "</think>"),
        "qwen35": ("<think>", "</think>"),
        "deepseek_r1": ("<think>", "</think>"),  # Same markers
        "nemotron": (" hesitation", " hesitation"),
    }

    start_token, end_token = markers.get(model_type, ("<think>", "</think>"))

    return ThinkingBudgetCriteria(
        tokenizer=tokenizer,
        thinking_budget=thinking_budget,
        thinking_start_token=start_token,
        thinking_end_token=end_token,
        enable_thinking=enable_thinking,
    )