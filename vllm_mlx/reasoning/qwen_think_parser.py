# SPDX-License-Identifier: Apache-2.0
"""
Unified reasoning parser for Qwen3 and Qwen3.5 models.

Qwen3 and Qwen3.5 use <think>...</think> tags for reasoning content.
They support implicit reasoning mode where <think> is injected in the prompt
by AI agents (e.g., OpenCode) and only </think> appears in the output.

This parser is a thin wrapper around BaseThinkingReasoningParser that defines
the specific tokens used by Qwen3/Qwen3.5 models.

Note: Both Qwen3 and Qwen3.5 use identical markers, so they share the same
parser implementation. Use the registry aliases ("qwen3", "qwen3.5", "qwen35")
to get this parser.
"""

from .think_parser import BaseThinkingReasoningParser


class QwenThinkParser(BaseThinkingReasoningParser):
    """
    Unified reasoning parser for Qwen3 and Qwen3.5 models.

    Qwen3/Qwen3.5 use <think>...</think> tokens to denote reasoning text.

    Supports three scenarios:
    1. Both tags in output: <think>reasoning</think>content
    2. Only closing tag (think in prompt): reasoning</think>content
    3. No tags: pure content

    Example (normal):
        Input: "<think>Let me analyze this...</think>The answer is 42."
        Output: reasoning="Let me analyze this...", content="The answer is 42."

    Example (think in prompt):
        Input: "Let me analyze this...</think>The answer is 42."
        Output: reasoning="Let me analyze this...", content="The answer is 42."
    """

    @property
    def start_token(self) -> str:
        """Return the start token for Qwen3/Qwen3.5 reasoning."""
        return "<think>"

    @property
    def end_token(self) -> str:
        """Return the end token for Qwen3/Qwen3.5 reasoning."""
        return "</think>"

    def extract_reasoning(
        self,
        model_output: str,
    ) -> tuple[str | None, str | None]:
        """
        Extract reasoning from Qwen3/Qwen3.5 output.

        Handles both explicit <think>...</think> tags and implicit mode
        where <think> was in the prompt (only </think> in output).

        Args:
            model_output: Complete model output text.

        Returns:
            (reasoning, content) tuple.
        """
        # If no end token at all, treat as pure content
        if self.end_token not in model_output:
            return None, model_output

        # Use base class implementation (handles both explicit and implicit)
        return super().extract_reasoning(model_output)