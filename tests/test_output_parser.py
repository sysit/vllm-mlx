# SPDX-License-Identifier: Apache-2.0
"""
Tests for ReasoningOutputParser.

Tests text-based and token-based reasoning extraction for Qwen models.
"""

import pytest

from vllm_mlx.reasoning.output_parser import (
    QWEN35_MARKERS,
    QWEN3_MARKERS,
    ReasoningOutputParser,
    create_qwen35_parser,
    create_qwen3_parser,
)


class TestReasoningOutputParserInit:
    """Test parser initialization."""

    def test_basic_init(self):
        """Test basic initialization without token IDs."""
        parser = ReasoningOutputParser(
            start_token="<think>",
            end_token="</think>",
        )
        assert parser.start_token == "<think>"
        assert parser.end_token == "</think>"
        assert parser.start_token_id is None
        assert parser.end_token_id is None

    def test_init_with_token_ids(self):
        """Test initialization with explicit token IDs."""
        parser = ReasoningOutputParser(
            start_token="<think>",
            end_token="</think>",
            start_token_id=151648,
            end_token_id=151649,
        )
        assert parser.start_token_id == 151648
        assert parser.end_token_id == 151649

    def test_factory_functions(self):
        """Test factory function parsers."""
        qwen3 = create_qwen3_parser()
        # Qwen3/Qwen3.5 use  hesita tags, not special Unicode markers
        assert qwen3.start_token == "<think>"
        assert qwen3.start_token_id is None  # Token IDs are NOT thinking markers

        qwen35 = create_qwen35_parser()
        assert qwen35.end_token_id is None  # Token IDs are NOT thinking markers


class TestNonStreamingExtraction:
    """Test extract_reasoning for complete outputs."""

    @pytest.fixture
    def parser(self):
        return create_qwen3_parser()

    def test_both_markers_present(self, parser):
        """Test extraction with both markers."""
        output = "<think>Let me think step by step</think>The answer is 42."
        reasoning, content = parser.extract_reasoning(output)
        assert reasoning == "Let me think step by step"
        assert content == "The answer is 42."

    def test_only_end_marker_implicit_mode(self, parser):
        """Test implicit reasoning mode (only end marker)."""
        output = "Thinking without start tag</think>Final answer"
        reasoning, content = parser.extract_reasoning(output)
        assert reasoning == "Thinking without start tag"
        assert content == "Final answer"

    def test_no_markers_pure_content(self, parser):
        """Test output without any markers."""
        output = "Just regular content here"
        reasoning, content = parser.extract_reasoning(output)
        assert reasoning is None
        assert content == "Just regular content here"

    def test_multiline_reasoning(self, parser):
        """Test reasoning with multiple lines."""
        output = "<think>\nStep 1: Analyze the problem\nStep 2: Compute result\n</think>42"
        reasoning, content = parser.extract_reasoning(output)
        assert "Step 1" in reasoning
        assert "Step 2" in reasoning
        assert content == "42"


class TestStreamingTextBased:
    """Test text-based streaming extraction."""

    @pytest.fixture
    def parser(self):
        parser = create_qwen3_parser()
        parser.reset_state()
        return parser

    def test_streaming_reasoning_phase(self, parser):
        """Test streaming during reasoning phase."""
        msg = parser.extract_reasoning_streaming("", "Let me think", "Let me think")
        assert msg is not None
        assert msg.reasoning == "Let me think"
        assert msg.content is None

    def test_streaming_content_phase(self, parser):
        """Test streaming after reasoning ends."""
        # First, process end marker
        parser.extract_reasoning_streaming(
            "reasoning</think>",
            "reasoning</think>content",
            "</think>content"
        )
        # Then pure content
        msg = parser.extract_reasoning_streaming(
            "reasoning</think>content",
            "reasoning</think>content here",
            " here"
        )
        assert msg.content == " here"

    def test_streaming_skip_markers(self, parser):
        """Test that pure marker deltas are skipped."""
        msg = parser.extract_reasoning_streaming("", "<think>", "<think>")
        # Should return None (marker skipped)
        assert msg is None

    def test_streaming_transition(self, parser):
        """Test transition from reasoning to content in one delta."""
        msg = parser.extract_reasoning_streaming(
            "reasoning",
            "reasoning</think>answer",
            "</think>answer"
        )
        assert msg is not None
        # Transition chunk: reasoning is empty (already in previous), content is "answer"
        assert msg.reasoning is None  # No new reasoning in this delta
        assert msg.content == "answer"


class TestStreamingTokenBased:
    """Test token-based streaming extraction (core feature)."""

    @pytest.fixture
    def parser(self):
        parser = create_qwen3_parser()
        parser.reset_state()
        return parser

    def test_token_based_reasoning_phase(self, parser):
        """Test token-based extraction during reasoning."""
        msg = parser.extract_streaming_with_tokens(
            delta_text="Let me think",
            delta_token_ids=[100, 101, 102]  # Random token IDs
        )
        assert msg is not None
        assert msg.reasoning == "Let me think"
        # end_token_id is None for Qwen3, so we check state differently
        assert not parser._in_content_phase

    def test_token_based_content_phase_after_end(self, parser):
        """Test behavior when token_ids are None (Qwen3/Qwen3.5 case).
        
        NOTE: When start_token_id and end_token_id are None,
        extract_streaming_with_tokens falls back to reasoning-only mode.
        For proper end marker detection, use extract_reasoning_streaming (text-based).
        """
        # Without token IDs, method treats all text as reasoning
        msg = parser.extract_streaming_with_tokens(
            delta_text="reasoning</think>",
            delta_token_ids=[100]
        )
        # Falls back to reasoning mode (no token-based detection)
        assert msg.reasoning == "reasoning</think>"
        
        # Use text-based method for proper detection
        parser.reset_state()
        parser.extract_reasoning_streaming("", "reasoning", "reasoning")
        msg = parser.extract_reasoning_streaming(
            "reasoning", "reasoning</think>Final answer", "</think>Final answer"
        )
        assert msg.content == "Final answer"

    def test_token_based_end_token_in_delta(self, parser):
        """Test end token detection behavior.
        
        NOTE: For Qwen3/Qwen3.5 (token_ids=None), use text-based method instead.
        """
        # Without token IDs, no token-based detection occurs
        msg = parser.extract_streaming_with_tokens(
            delta_text="reasoning</think>content",
            delta_token_ids=[100, 101]
        )
        # Falls back to reasoning-only mode
        assert msg.reasoning == "reasoning</think>content"
        
        # Use text-based method for proper detection
        parser.reset_state()
        msg = parser.extract_reasoning_streaming(
            "", "reasoning</think>content", "reasoning</think>content"
        )
        assert "reasoning" in msg.reasoning
        assert "content" in msg.content

    def test_token_based_start_token_stripping(self, parser):
        """Test start token stripping behavior.
        
        NOTE: For Qwen3/Qwen3.5 (token_ids=None), start token is NOT stripped
        in extract_streaming_with_tokens. Use extract_reasoning_streaming instead.
        """
        # Without token IDs, start token is NOT stripped
        msg = parser.extract_streaming_with_tokens(
            delta_text="<think>reasoning text",
            delta_token_ids=[100, 101]
        )
        # Start token remains in reasoning (no token-based stripping)
        assert "<think>" in msg.reasoning
        
        # Use text-based method for proper handling
        parser.reset_state()
        msg = parser.extract_reasoning_streaming("", "<think>reasoning text", "<think>reasoning text")
        # Text-based method strips start token
        assert "<think>" not in msg.reasoning

    def test_token_based_multi_token_delta(self, parser):
        """Test handling of multi-token deltas.
        
        NOTE: For Qwen3/Qwen3.5 (token_ids=None), use text-based method instead.
        """
        delta_text = "final reasoning</think>answer"
        delta_ids = [100, 101]
        
        msg = parser.extract_streaming_with_tokens(
            delta_text=delta_text,
            delta_token_ids=delta_ids
        )
        # Without token IDs, falls back to reasoning-only mode
        assert "final reasoning</think>answer" in msg.reasoning
        
        # Use text-based method for proper detection
        parser.reset_state()
        msg = parser.extract_reasoning_streaming(
            "", delta_text, delta_text
        )
        assert "final reasoning" in msg.reasoning
        assert "answer" in msg.content
        assert parser._in_content_phase

    def test_token_based_state_tracking(self, parser):
        """Test that previous_token_ids are tracked."""
        parser.extract_streaming_with_tokens("text1", [100, 101])
        assert 100 in parser._previous_token_ids
        assert 101 in parser._previous_token_ids
        
        parser.extract_streaming_with_tokens("text2", [102])
        assert 102 in parser._previous_token_ids
        assert len(parser._previous_token_ids) == 3


class TestResetState:
    """Test state reset functionality."""

    @pytest.fixture
    def parser(self):
        parser = create_qwen3_parser()
        # Simulate some state
        parser._previous_token_ids = [1, 2, 3]
        parser._in_content_phase = True
        return parser

    def test_reset_clears_token_ids(self, parser):
        """Test that reset clears previous_token_ids."""
        parser.reset_state()
        assert parser._previous_token_ids == []

    def test_reset_clears_phase_state(self, parser):
        """Test that reset clears in_content_phase."""
        parser.reset_state()
        assert parser._in_content_phase is False


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    @pytest.fixture
    def parser(self):
        parser = create_qwen3_parser()
        parser.reset_state()
        return parser

    def test_empty_delta(self, parser):
        """Test empty delta text."""
        msg = parser.extract_streaming_with_tokens("", [])
        assert msg is None  # Empty delta should return None

    def test_whitespace_only_reasoning(self, parser):
        """Test reasoning with only whitespace."""
        output = "<think>   </think>content"
        reasoning, content = parser.extract_reasoning(output)
        # Whitespace-only reasoning should be None
        assert reasoning is None or reasoning.strip() == ""

    def test_marker_at_boundary(self, parser):
        """Test marker at exact boundary positions."""
        # End marker at start of content
        output = "<think>think</think>"
        reasoning, content = parser.extract_reasoning(output)
        assert reasoning == "think"
        assert content is None or content == ""

    def test_no_token_ids_graceful_fallback(self):
        """Test graceful fallback when token IDs not available."""
        parser = ReasoningOutputParser(
            start_token="<think>",
            end_token="</think>",
            # No token IDs provided
        )
        parser.reset_state()
        
        # Should still work with text-based approach
        msg = parser.extract_streaming_with_tokens("reasoning text", [100, 101])
        assert msg is not None
        assert msg.reasoning == "reasoning text"


class TestIntegrationWithTextBasedParser:
    """Test integration between token-based and text-based methods."""

    @pytest.fixture
    def parser(self):
        parser = create_qwen3_parser()
        parser.reset_state()
        return parser

    def test_text_based_fallback_for_no_token_ids(self, parser):
        """Test that text-based method still works."""
        # Use text-based streaming (no token IDs)
        msg = parser.extract_reasoning_streaming(
            previous_text="",
            current_text="reasoning",
            delta_text="reasoning"
        )
        assert msg is not None
        assert msg.reasoning == "reasoning"

    def test_both_methods_consistent(self, parser):
        """Test that both methods produce consistent results."""
        # Token-based
        parser.reset_state()
        msg1 = parser.extract_streaming_with_tokens("reasoning", [100])
        
        # Text-based
        parser.reset_state()
        msg2 = parser.extract_reasoning_streaming("", "reasoning", "reasoning")
        
        # Both should mark as reasoning
        assert msg1.reasoning == msg2.reasoning


class TestMarkerConfigurations:
    """Test marker configuration constants."""

    def test_qwen3_markers(self):
        """Test Qwen3 marker config."""
        # Qwen3 uses  hesita tags, not special Unicode markers
        assert QWEN3_MARKERS["start_token"] == "<think>"
        assert QWEN3_MARKERS["end_token"] == "</think>"
        # Token IDs are None because 151648/151649 are NOT thinking markers
        assert QWEN3_MARKERS["start_token_id"] is None
        assert QWEN3_MARKERS["end_token_id"] is None

    def test_qwen35_markers(self):
        """Test Qwen3.5 marker config."""
        # Qwen3.5 uses same markers as Qwen3
        assert QWEN35_MARKERS["start_token"] == "<think>"
        assert QWEN35_MARKERS["end_token"] == "</think>"
        # Token IDs are None because 151648/151649 are NOT thinking markers
        assert QWEN35_MARKERS["start_token_id"] is None
        assert QWEN35_MARKERS["end_token_id"] is None
