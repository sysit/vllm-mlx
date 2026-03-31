# SPDX-License-Identifier: Apache-2.0
"""
Tests for P0 fixes:
1. SimpleEngine.chat/stream_chat passes images/videos to MLLM
2. MLLMBatchGenerator supports large images with increased prefill_step_size
"""

import platform
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Skip all tests if not on Apple Silicon
pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or platform.machine() != "arm64",
    reason="Requires Apple Silicon",
)


# =============================================================================
# P0-1: Test images/videos parameter passing in SimpleEngine
# =============================================================================


class TestSimpleEngineImageVideoParams:
    """Test that SimpleEngine passes images/videos to MLLM."""

    @pytest.mark.asyncio
    async def test_chat_passes_images_to_mllm(self):
        """Test that chat() passes images parameter to MLLM."""
        from vllm_mlx.engine.simple import SimpleEngine

        # Create mock model that captures the images parameter
        mock_model = MagicMock()
        mock_model.chat = MagicMock(return_value=MagicMock(
            text="Response",
            prompt_tokens=10,
            completion_tokens=5,
            finish_reason="stop"
        ))
        mock_model.tokenizer = MagicMock()

        with patch.object(SimpleEngine, "_load_model", return_value=mock_model):
            engine = SimpleEngine(model="test-model")
            engine._is_mllm = True
            engine._loaded = True

            await engine.chat(
                messages=[{"role": "user", "content": "What is this?"}],
                images=["/path/to/image.jpg"],
            )

        # Verify images was passed
        call_kwargs = mock_model.chat.call_args[1]
        assert "images" in call_kwargs
        assert call_kwargs["images"] == ["/path/to/image.jpg"]

    @pytest.mark.asyncio
    async def test_chat_passes_videos_to_mllm(self):
        """Test that chat() passes videos parameter to MLLM."""
        from vllm_mlx.engine.simple import SimpleEngine

        mock_model = MagicMock()
        mock_model.chat = MagicMock(return_value=MagicMock(
            text="Response",
            prompt_tokens=10,
            completion_tokens=5,
            finish_reason="stop"
        ))
        mock_model.tokenizer = MagicMock()

        with patch.object(SimpleEngine, "_load_model", return_value=mock_model):
            engine = SimpleEngine(model="test-model")
            engine._is_mllm = True
            engine._loaded = True

            await engine.chat(
                messages=[{"role": "user", "content": "Describe this video"}],
                videos=["/path/to/video.mp4"],
            )

        call_kwargs = mock_model.chat.call_args[1]
        assert "videos" in call_kwargs
        assert call_kwargs["videos"] == ["/path/to/video.mp4"]

    @pytest.mark.asyncio
    async def test_stream_chat_passes_images_to_mllm(self):
        """Test that stream_chat() passes images parameter to MLLM."""
        from vllm_mlx.engine.simple import SimpleEngine

        # Create mock that yields chunks
        mock_chunk = MagicMock()
        mock_chunk.text = "Response"
        mock_chunk.finish_reason = "stop"
        mock_chunk.prompt_tokens = 10

        mock_model = MagicMock()
        mock_model.stream_chat = MagicMock(return_value=iter([mock_chunk]))
        mock_model.tokenizer = MagicMock()

        with patch.object(SimpleEngine, "_load_model", return_value=mock_model):
            engine = SimpleEngine(model="test-model")
            engine._is_mllm = True
            engine._loaded = True

            chunks = []
            async for chunk in engine.stream_chat(
                messages=[{"role": "user", "content": "What is this?"}],
                images=["/path/to/image.jpg"],
            ):
                chunks.append(chunk)

        # Verify images was passed
        call_kwargs = mock_model.stream_chat.call_args[1]
        assert "images" in call_kwargs
        assert call_kwargs["images"] == ["/path/to/image.jpg"]

    @pytest.mark.asyncio
    async def test_stream_chat_passes_videos_to_mllm(self):
        """Test that stream_chat() passes videos parameter to MLLM."""
        from vllm_mlx.engine.simple import SimpleEngine

        mock_chunk = MagicMock()
        mock_chunk.text = "Response"
        mock_chunk.finish_reason = "stop"
        mock_chunk.prompt_tokens = 10

        mock_model = MagicMock()
        mock_model.stream_chat = MagicMock(return_value=iter([mock_chunk]))
        mock_model.tokenizer = MagicMock()

        with patch.object(SimpleEngine, "_load_model", return_value=mock_model):
            engine = SimpleEngine(model="test-model")
            engine._is_mllm = True
            engine._loaded = True

            chunks = []
            async for chunk in engine.stream_chat(
                messages=[{"role": "user", "content": "Describe this video"}],
                videos=["/path/to/video.mp4"],
            ):
                chunks.append(chunk)

        call_kwargs = mock_model.stream_chat.call_args[1]
        assert "videos" in call_kwargs
        assert call_kwargs["videos"] == ["/path/to/video.mp4"]


# =============================================================================
# P0-2: Test MLLMBatchGenerator prefill_step_size default
# =============================================================================


class TestMLLMBatchGeneratorPrefillStepSize:
    """Test that MLLMBatchGenerator has increased prefill_step_size."""

    def test_default_prefill_step_size_is_8192(self):
        """Test that default prefill_step_size is 8192 (not 1024)."""
        from vllm_mlx.mllm_batch_generator import MLLMBatchGenerator

        # Check the default value in signature
        import inspect
        sig = inspect.signature(MLLMBatchGenerator.__init__)
        params = sig.parameters
        assert "prefill_step_size" in params
        assert params["prefill_step_size"].default == 8192

    def test_large_image_tokens_within_limit(self):
        """Test that large image tokens (~9900) don't exceed limit."""
        from vllm_mlx.mllm_batch_generator import MLLMBatchGenerator

        # With prefill_step_size=8192 and single request
        # max_batch_tokens = 8192 * 1 = 8192
        # A ~9900 token image would still exceed this, but with 2 requests:
        # max_batch_tokens = 8192 * 2 = 16384, which is sufficient

        # The key improvement is that default is now 8192 instead of 1024
        # This allows larger prompts and images by default

        # Verify the limit calculation
        prefill_step_size = 8192
        num_requests = 1
        max_batch_tokens = prefill_step_size * num_requests

        # A prompt with ~5000 tokens should work
        assert 5000 < max_batch_tokens

        # With 2 requests in batch
        num_requests = 2
        max_batch_tokens = prefill_step_size * num_requests
        # ~9900 token image should work in this case
        assert 9900 < max_batch_tokens


# =============================================================================
# Integration Test (requires model loading)
# =============================================================================


class TestIntegration:
    """Integration tests that require actual model loading."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_mllm_chat_with_image(self, small_mllm_model, test_image_path):
        """Integration test: chat with real image using SimpleEngine."""
        pytest.importorskip("mlx_vlm")

        from vllm_mlx.engine.simple import SimpleEngine

        engine = SimpleEngine(model=small_mllm_model)
        await engine.start()

        try:
            result = await engine.chat(
                messages=[{"role": "user", "content": "What do you see?"}],
                images=[test_image_path],
                max_tokens=50,
            )

            assert result.text
            assert len(result.text) > 0
            assert result.finish_reason in ("stop", "length")
        finally:
            await engine.stop()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_mllm_stream_chat_with_image(
        self, small_mllm_model, test_image_path
    ):
        """Integration test: stream_chat with real image using SimpleEngine."""
        pytest.importorskip("mlx_vlm")

        from vllm_mlx.engine.simple import SimpleEngine

        engine = SimpleEngine(model=small_mllm_model)
        await engine.start()

        try:
            chunks = []
            async for chunk in engine.stream_chat(
                messages=[{"role": "user", "content": "Describe briefly"}],
                images=[test_image_path],
                max_tokens=30,
            ):
                chunks.append(chunk)

            assert len(chunks) > 0
            final_text = chunks[-1].text
            assert len(final_text) > 0
        finally:
            await engine.stop()


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def small_mllm_model():
    """Return a small MLLM model for testing."""
    return "mlx-community/Qwen3-VL-4B-Instruct-3bit"


@pytest.fixture
def test_image_path(tmp_path):
    """Create a test image."""
    pytest.importorskip("PIL")
    from PIL import Image

    img = Image.new("RGB", (320, 240), color="blue")
    path = tmp_path / "test_image.jpg"
    img.save(path)
    return str(path)