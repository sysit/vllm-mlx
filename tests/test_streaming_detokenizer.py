# SPDX-License-Identifier: Apache-2.0
"""Tests for streaming detokenizer optimization in scheduler."""

import pytest
from unittest.mock import MagicMock
from transformers import AutoTokenizer
from mlx_lm.tokenizer_utils import (
    NaiveStreamingDetokenizer,
    BPEStreamingDetokenizer,
)


class MockTokenizer:
    """A simple mock tokenizer for testing without network dependency."""
    
    def __init__(self):
        # Simple vocab for testing: maps tokens to chars/words
        self._vocab = {
            0: "",
            1: "Hello",
            2: ",",
            3: " ",
            4: "how",
            5: " are",
            6: " you",
            7: " doing",
            8: " today",
            9: "?",
            10: "!",
            11: "I",
            12: " hope",
            13: " you're",
            14: " well",
            15: "world",
            16: "test",
            17: " 你",
            18: "好",
            19: " م",
            20: "ر",
            21: "ح",
            22: "ب",
            23: "ا",
            24: " 🎉",
            25: "Hi",
            26: "Goodbye",
            27: "Test",
            28: "message",
        }
        self._reverse_vocab = {v: k for k, v in self._vocab.items()}
        self.clean_up_tokenization_spaces = True
    
    def encode(self, text):
        """Simple mock encoding - finds matching tokens in text."""
        tokens = []
        remaining = text
        # Sort by length to match longer strings first
        for word in sorted(self._vocab.values(), key=len, reverse=True):
            if word and word in remaining:
                idx = self._reverse_vocab.get(word)
                if idx is not None:
                    tokens.append(idx)
                    remaining = remaining.replace(word, "", 1)
        # If no tokens found, return a default token for testing
        if not tokens and text:
            tokens = [1]  # Default to "Hello" token for unknown text
        return tokens
    
    def decode(self, tokens):
        """Decode tokens to text."""
        if isinstance(tokens, int):
            tokens = [tokens]
        result = ""
        for t in tokens:
            if t in self._vocab:
                result += self._vocab[t]
        return result


class TestStreamingDetokenizer:
    """Test streaming detokenizer correctness."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Provide a mock tokenizer for fast tests without network."""
        return MockTokenizer()

    @pytest.fixture
    def qwen_tokenizer(self, request):
        """Load Qwen tokenizer (requires --run-slow and network)."""
        if not request.config.getoption("--run-slow"):
            # Return mock tokenizer for fast tests
            return MockTokenizer()
        return AutoTokenizer.from_pretrained("mlx-community/Qwen3-0.6B-8bit")

    @pytest.mark.slow
    def test_naive_streaming_matches_batch(self, qwen_tokenizer):
        """Verify NaiveStreamingDetokenizer output matches batch decode (requires network)."""
        text = "Hello, how are you doing today? I hope you're well!"
        tokens = qwen_tokenizer.encode(text)

        # Streaming decode
        detok = NaiveStreamingDetokenizer(qwen_tokenizer)
        detok.reset()
        for t in tokens:
            detok.add_token(t)
        detok.finalize()
        streaming_result = detok.text

        # Batch decode
        batch_result = qwen_tokenizer.decode(tokens)

        assert streaming_result == batch_result, (
            f"Streaming: {repr(streaming_result)}\n" f"Batch: {repr(batch_result)}"
        )

    def test_mock_tokenizer_streaming(self, mock_tokenizer):
        """Test streaming detokenizer with mock tokenizer (no network required)."""
        # Use simple token sequence
        tokens = [1, 2, 3, 4]  # "Hello", ",", " ", "how"

        detok = NaiveStreamingDetokenizer(mock_tokenizer)
        detok.reset()
        for t in tokens:
            detok.add_token(t)
        detok.finalize()

        streaming_result = detok.text
        batch_result = mock_tokenizer.decode(tokens)

        assert streaming_result == batch_result, (
            f"Streaming: {repr(streaming_result)}\n" f"Batch: {repr(batch_result)}"
        )

    def test_last_segment_incremental(self, qwen_tokenizer):
        """Verify last_segment returns only new text."""
        text = "Hello world"
        tokens = qwen_tokenizer.encode(text)

        detok = NaiveStreamingDetokenizer(qwen_tokenizer)
        detok.reset()

        segments = []
        for t in tokens:
            detok.add_token(t)
            seg = detok.last_segment
            if seg:
                segments.append(seg)

        detok.finalize()

        # Concatenated segments should equal full text
        full_from_segments = "".join(segments) + detok.last_segment
        assert full_from_segments == detok.text

    def test_reset_clears_state(self, qwen_tokenizer):
        """Verify reset() clears all state."""
        detok = NaiveStreamingDetokenizer(qwen_tokenizer)

        # Add some tokens
        tokens = qwen_tokenizer.encode("Hello")
        for t in tokens:
            detok.add_token(t)

        # Reset
        detok.reset()

        # State should be cleared
        assert detok.tokens == []
        assert detok.offset == 0

    def test_unicode_handling(self, qwen_tokenizer):
        """Test handling of unicode characters."""
        text = "Hello 你好 مرحبا 🎉"
        tokens = qwen_tokenizer.encode(text)

        detok = NaiveStreamingDetokenizer(qwen_tokenizer)
        detok.reset()
        for t in tokens:
            detok.add_token(t)
        detok.finalize()

        batch_result = qwen_tokenizer.decode(tokens)
        assert detok.text == batch_result

    def test_empty_input(self, qwen_tokenizer):
        """Test with empty input."""
        detok = NaiveStreamingDetokenizer(qwen_tokenizer)
        detok.reset()
        detok.finalize()

        assert detok.text == ""
        assert detok.tokens == []

    def test_single_token(self, qwen_tokenizer):
        """Test with single token."""
        tokens = [qwen_tokenizer.encode("Hi")[0]]

        detok = NaiveStreamingDetokenizer(qwen_tokenizer)
        detok.reset()
        detok.add_token(tokens[0])
        detok.finalize()

        batch_result = qwen_tokenizer.decode(tokens)
        assert detok.text == batch_result


class TestSchedulerDetokenizer:
    """Test scheduler's detokenizer integration."""

    @pytest.fixture
    def scheduler_mock(self, request):
        """Create a mock scheduler with detokenizer pool."""
        if not request.config.getoption("--run-slow"):
            # Use mock tokenizer for fast tests
            class MockScheduler:
                def __init__(self):
                    self.tokenizer = MockTokenizer()
                    self._detokenizer_pool = {}

                def _get_detokenizer(self, request_id):
                    if request_id not in self._detokenizer_pool:
                        detok = NaiveStreamingDetokenizer(self.tokenizer)
                        detok.reset()
                        self._detokenizer_pool[request_id] = detok
                    return self._detokenizer_pool[request_id]

                def _cleanup_detokenizer(self, request_id):
                    if request_id in self._detokenizer_pool:
                        del self._detokenizer_pool[request_id]

            return MockScheduler()

        # Load real tokenizer for slow tests (requires network)
        from transformers import AutoTokenizer

        class MockScheduler:
            def __init__(self):
                self.tokenizer = AutoTokenizer.from_pretrained(
                    "mlx-community/Qwen3-0.6B-8bit"
                )
                self._detokenizer_pool = {}

            def _get_detokenizer(self, request_id):
                if request_id not in self._detokenizer_pool:
                    detok = NaiveStreamingDetokenizer(self.tokenizer)
                    detok.reset()
                    self._detokenizer_pool[request_id] = detok
                return self._detokenizer_pool[request_id]

            def _cleanup_detokenizer(self, request_id):
                if request_id in self._detokenizer_pool:
                    del self._detokenizer_pool[request_id]

        return MockScheduler()

    def test_detokenizer_pool_creation(self, scheduler_mock):
        """Test that detokenizers are created on demand."""
        assert len(scheduler_mock._detokenizer_pool) == 0

        detok1 = scheduler_mock._get_detokenizer("req1")
        assert len(scheduler_mock._detokenizer_pool) == 1

        detok2 = scheduler_mock._get_detokenizer("req2")
        assert len(scheduler_mock._detokenizer_pool) == 2

        # Same request ID returns same detokenizer
        detok1_again = scheduler_mock._get_detokenizer("req1")
        assert detok1 is detok1_again

    def test_detokenizer_cleanup(self, scheduler_mock):
        """Test that cleanup removes detokenizers."""
        scheduler_mock._get_detokenizer("req1")
        scheduler_mock._get_detokenizer("req2")
        assert len(scheduler_mock._detokenizer_pool) == 2

        scheduler_mock._cleanup_detokenizer("req1")
        assert len(scheduler_mock._detokenizer_pool) == 1
        assert "req1" not in scheduler_mock._detokenizer_pool

        scheduler_mock._cleanup_detokenizer("req2")
        assert len(scheduler_mock._detokenizer_pool) == 0

    def test_cleanup_nonexistent_is_safe(self, scheduler_mock):
        """Test that cleaning up nonexistent request doesn't raise."""
        scheduler_mock._cleanup_detokenizer("nonexistent")  # Should not raise

    def test_multiple_requests_independent(self, scheduler_mock):
        """Test that multiple requests have independent detokenizers."""
        detok1 = scheduler_mock._get_detokenizer("req1")
        detok2 = scheduler_mock._get_detokenizer("req2")

        # Add different tokens to each
        tokens1 = scheduler_mock.tokenizer.encode("Hello")
        tokens2 = scheduler_mock.tokenizer.encode("Goodbye")

        for t in tokens1:
            detok1.add_token(t)
        for t in tokens2:
            detok2.add_token(t)

        detok1.finalize()
        detok2.finalize()

        # Results should be independent
        assert "Hello" in detok1.text
        assert "Goodbye" in detok2.text
        assert detok1.text != detok2.text


@pytest.mark.slow
class TestOptimizedDetokenizer:
    """Test that optimized detokenizer is used when available (requires network)."""

    @pytest.fixture
    def tokenizer_wrapper(self):
        """Load tokenizer via mlx_lm to get TokenizerWrapper with optimized detokenizer."""
        from mlx_lm import load

        _, tokenizer = load("mlx-community/Qwen3-0.6B-8bit")
        return tokenizer

    def test_tokenizer_wrapper_has_optimized_detokenizer(self, tokenizer_wrapper):
        """Verify TokenizerWrapper has optimized detokenizer class."""
        assert hasattr(tokenizer_wrapper, "_detokenizer_class")
        assert hasattr(tokenizer_wrapper, "detokenizer")
        # Qwen uses BPE tokenizer
        assert tokenizer_wrapper._detokenizer_class == BPEStreamingDetokenizer

    def test_optimized_detokenizer_correctness(self, tokenizer_wrapper):
        """Verify optimized detokenizer produces correct output."""
        text = "Hello, how are you doing today?"
        raw_tokenizer = tokenizer_wrapper._tokenizer
        tokens = raw_tokenizer.encode(text)

        # Use optimized detokenizer
        detok = tokenizer_wrapper.detokenizer
        detok.reset()
        for t in tokens:
            detok.add_token(t)
        detok.finalize()

        # Compare with batch decode
        batch_result = raw_tokenizer.decode(tokens)
        assert detok.text == batch_result

    def test_scheduler_uses_optimized_detokenizer(self, tokenizer_wrapper):
        """Test that scheduler-like code uses optimized detokenizer."""
        # Simulate scheduler's _get_detokenizer logic
        _detokenizer_pool = {}

        def _get_detokenizer(tokenizer, request_id):
            if request_id not in _detokenizer_pool:
                if hasattr(tokenizer, "detokenizer"):
                    detok = tokenizer.detokenizer
                else:
                    detok = NaiveStreamingDetokenizer(tokenizer)
                detok.reset()
                _detokenizer_pool[request_id] = detok
            return _detokenizer_pool[request_id]

        # Get detokenizer - should be BPEStreamingDetokenizer
        detok = _get_detokenizer(tokenizer_wrapper, "test_req")
        assert isinstance(detok, BPEStreamingDetokenizer)

        # Verify it works correctly
        raw_tokenizer = tokenizer_wrapper._tokenizer
        tokens = raw_tokenizer.encode("Test message")
        for t in tokens:
            detok.add_token(t)
        detok.finalize()
        assert detok.text == raw_tokenizer.decode(tokens)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
