# SPDX-License-Identifier: Apache-2.0
"""
Phase 2: API Layer Integration Tests.

Tests for:
1. OpenAI streaming response builder with reasoning field
2. enable_thinking parameter passing
3. Non-streaming response formatting

Reference: vllm-mlx Phase 2 design
"""

import json
import pytest

from vllm_mlx.api.streaming import StreamingJSONEncoder
from vllm_mlx.api.models import (
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    AssistantMessage,
    ChatCompletionChoice,
    ChatCompletionResponse,
    Usage,
)
from vllm_mlx.reasoning import (
    DeltaMessage,
    create_qwen3_parser,
)


class TestStreamingJSONEncoderReasoning:
    """Test StreamingJSONEncoder with reasoning field."""

    def test_encode_chat_chunk_with_reasoning_only(self):
        """Test chunk with only reasoning content."""
        encoder = StreamingJSONEncoder(
            response_id="chatcmpl-test",
            model="qwen3-4b",
            object_type="chat.completion.chunk",
        )

        chunk = encoder.encode_chat_chunk(reasoning="Let me think...")

        # Parse JSON to verify structure
        lines = chunk.strip().split("\n")
        assert lines[0].startswith("data: ")
        data_json = lines[0][6:]  # Strip "data: "
        parsed = json.loads(data_json)

        assert parsed["id"] == "chatcmpl-test"
        assert parsed["model"] == "qwen3-4b"
        assert parsed["object"] == "chat.completion.chunk"
        assert len(parsed["choices"]) == 1

        delta = parsed["choices"][0]["delta"]
        assert "reasoning" in delta
        assert delta["reasoning"] == "Let me think..."
        assert "content" not in delta or delta["content"] is None

    def test_encode_chat_chunk_with_content_only(self):
        """Test chunk with only content (standard response)."""
        encoder = StreamingJSONEncoder(
            response_id="chatcmpl-test",
            model="qwen3-4b",
            object_type="chat.completion.chunk",
        )

        chunk = encoder.encode_chat_chunk(content="The answer is 42.")

        lines = chunk.strip().split("\n")
        data_json = lines[0][6:]
        parsed = json.loads(data_json)

        delta = parsed["choices"][0]["delta"]
        assert "content" in delta
        assert delta["content"] == "The answer is 42."
        assert "reasoning" not in delta or delta["reasoning"] is None

    def test_encode_chat_chunk_with_both_reasoning_and_content(self):
        """Test chunk with both reasoning and content (transition chunk)."""
        encoder = StreamingJSONEncoder(
            response_id="chatcmpl-test",
            model="qwen3-4b",
            object_type="chat.completion.chunk",
        )

        # Transition chunk: final reasoning + first content
        chunk = encoder.encode_chat_chunk(
            reasoning="final step",
            content="Final answer",
        )

        lines = chunk.strip().split("\n")
        data_json = lines[0][6:]
        parsed = json.loads(data_json)

        delta = parsed["choices"][0]["delta"]
        assert delta["reasoning"] == "final step"
        assert delta["content"] == "Final answer"

    def test_encode_chat_chunk_with_role_and_reasoning(self):
        """Test first chunk with role and reasoning."""
        encoder = StreamingJSONEncoder(
            response_id="chatcmpl-test",
            model="qwen3-4b",
            object_type="chat.completion.chunk",
        )

        chunk = encoder.encode_chat_chunk(role="assistant", reasoning="Thinking...")

        lines = chunk.strip().split("\n")
        data_json = lines[0][6:]
        parsed = json.loads(data_json)

        delta = parsed["choices"][0]["delta"]
        assert delta["role"] == "assistant"
        assert delta["reasoning"] == "Thinking..."

    def test_encode_chat_chunk_with_finish_reason(self):
        """Test final chunk with finish_reason."""
        encoder = StreamingJSONEncoder(
            response_id="chatcmpl-test",
            model="qwen3-4b",
            object_type="chat.completion.chunk",
        )

        chunk = encoder.encode_chat_chunk(
            content="Done",
            finish_reason="stop",
        )

        lines = chunk.strip().split("\n")
        data_json = lines[0][6:]
        parsed = json.loads(data_json)

        assert parsed["choices"][0]["finish_reason"] == "stop"

    def test_encode_chat_chunk_with_usage(self):
        """Test final chunk with usage stats."""
        encoder = StreamingJSONEncoder(
            response_id="chatcmpl-test",
            model="qwen3-4b",
            object_type="chat.completion.chunk",
        )

        usage = {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        }
        chunk = encoder.encode_chat_chunk(
            content="Final",
            finish_reason="stop",
            usage=usage,
        )

        lines = chunk.strip().split("\n")
        data_json = lines[0][6:]
        parsed = json.loads(data_json)

        assert parsed["usage"] == usage

    def test_encode_done(self):
        """Test [DONE] message."""
        encoder = StreamingJSONEncoder(
            response_id="chatcmpl-test",
            model="qwen3-4b",
            object_type="chat.completion.chunk",
        )

        done = encoder.encode_done()
        assert done == "data: [DONE]\n\n"


class TestChatCompletionChunkWithReasoning:
    """Test Pydantic ChatCompletionChunk with reasoning field."""

    def test_chunk_with_reasoning(self):
        """Test ChatCompletionChunk with reasoning in delta."""
        chunk = ChatCompletionChunk(
            id="chatcmpl-test",
            model="qwen3-4b",
            choices=[
                ChatCompletionChunkChoice(
                    delta=ChatCompletionChunkDelta(reasoning="Thinking..."),
                )
            ],
        )

        # Serialize to JSON
        json_str = chunk.model_dump_json()
        parsed = json.loads(json_str)

        assert parsed["choices"][0]["delta"]["reasoning"] == "Thinking..."

    def test_chunk_with_both_fields(self):
        """Test ChatCompletionChunk with both reasoning and content."""
        chunk = ChatCompletionChunk(
            id="chatcmpl-test",
            model="qwen3-4b",
            choices=[
                ChatCompletionChunkChoice(
                    delta=ChatCompletionChunkDelta(
                        reasoning="Final reasoning",
                        content="The answer",
                    ),
                )
            ],
        )

        json_str = chunk.model_dump_json()
        parsed = json.loads(json_str)

        delta = parsed["choices"][0]["delta"]
        assert delta["reasoning"] == "Final reasoning"
        assert delta["content"] == "The answer"

    def test_chunk_with_finish_reason_and_usage(self):
        """Test final chunk with finish_reason and usage."""
        chunk = ChatCompletionChunk(
            id="chatcmpl-test",
            model="qwen3-4b",
            choices=[
                ChatCompletionChunkChoice(
                    delta=ChatCompletionChunkDelta(content="Done"),
                    finish_reason="stop",
                )
            ],
            usage=Usage(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
            ),
        )

        json_str = chunk.model_dump_json()
        parsed = json.loads(json_str)

        assert parsed["choices"][0]["finish_reason"] == "stop"
        assert parsed["usage"]["total_tokens"] == 30


class TestAssistantMessageWithReasoning:
    """Test AssistantMessage with reasoning field."""

    def test_message_with_reasoning_only(self):
        """Test AssistantMessage with reasoning only."""
        msg = AssistantMessage(reasoning="Deep thought process...")

        json_str = msg.model_dump_json()
        parsed = json.loads(json_str)

        assert parsed["role"] == "assistant"
        assert parsed["reasoning"] == "Deep thought process..."
        # reasoning_content alias should also be present
        assert parsed["reasoning_content"] == "Deep thought process..."

    def test_message_with_both_reasoning_and_content(self):
        """Test AssistantMessage with both reasoning and content."""
        msg = AssistantMessage(
            reasoning="Step-by-step analysis",
            content="The result is 42",
        )

        json_str = msg.model_dump_json()
        parsed = json.loads(json_str)

        assert parsed["reasoning"] == "Step-by-step analysis"
        assert parsed["content"] == "The result is 42"

    def test_message_with_content_only(self):
        """Test AssistantMessage with content only (standard response)."""
        msg = AssistantMessage(content="Standard response")

        json_str = msg.model_dump_json()
        parsed = json.loads(json_str)

        assert parsed["content"] == "Standard response"
        assert parsed.get("reasoning") is None


class TestChatCompletionResponseWithReasoning:
    """Test ChatCompletionResponse with reasoning."""

    def test_response_with_reasoning(self):
        """Test complete ChatCompletionResponse with reasoning."""
        response = ChatCompletionResponse(
            model="qwen3-4b",
            choices=[
                ChatCompletionChoice(
                    message=AssistantMessage(
                        reasoning="Let me analyze this...",
                        content="The answer is 42",
                    ),
                    finish_reason="stop",
                )
            ],
            usage=Usage(
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
            ),
        )

        json_str = response.model_dump_json()
        parsed = json.loads(json_str)

        assert parsed["object"] == "chat.completion"
        msg = parsed["choices"][0]["message"]
        assert msg["reasoning"] == "Let me analyze this..."
        assert msg["content"] == "The answer is 42"


class TestReasoningParserIntegrationWithAPI:
    """Test integration between ReasoningOutputParser and API models."""

    @pytest.fixture
    def parser(self):
        parser = create_qwen3_parser()
        parser.reset_state()
        return parser

    def test_parser_delta_to_chunk_delta(self, parser):
        """Test converting parser DeltaMessage to ChatCompletionChunkDelta."""
        # Simulate reasoning phase
        delta_msg = parser.extract_streaming_with_tokens(
            delta_text="Let me think",
            delta_token_ids=[100, 101, 102],
        )

        assert delta_msg is not None
        assert delta_msg.reasoning == "Let me think"

        # Convert to API delta
        api_delta = ChatCompletionChunkDelta(
            reasoning=delta_msg.reasoning,
            content=delta_msg.content,
        )

        json_str = api_delta.model_dump_json()
        parsed = json.loads(json_str)

        assert parsed["reasoning"] == "Let me think"
        assert parsed.get("content") is None

    def test_parser_transition_chunk(self, parser):
        """Test transition from reasoning to content."""
        # End marker in delta
        delta_msg = parser.extract_streaming_with_tokens(
            delta_text="final reasoning<｜end▁of▁thinking▁｜>answer",
            delta_token_ids=[100, 151649, 200],
        )

        assert delta_msg is not None
        # Should have both reasoning and content
        assert "reasoning" in delta_msg.reasoning
        assert "answer" in delta_msg.content

        # Convert to API delta
        api_delta = ChatCompletionChunkDelta(
            reasoning=delta_msg.reasoning,
            content=delta_msg.content,
        )

        json_str = api_delta.model_dump_json()
        parsed = json.loads(json_str)

        # Both fields present in transition chunk
        assert parsed["reasoning"] is not None
        assert parsed["content"] is not None

    def test_parser_content_phase_chunk(self, parser):
        """Test content phase after reasoning ends."""
        # First, process end marker
        parser.extract_streaming_with_tokens(
            delta_text="<｜end▁of▁thinking▁｜>",
            delta_token_ids=[151649],
        )

        # Now pure content
        delta_msg = parser.extract_streaming_with_tokens(
            delta_text="Final answer",
            delta_token_ids=[200, 201, 202],
        )

        assert delta_msg.content == "Final answer"
        assert delta_msg.reasoning is None

        # Convert to API delta
        api_delta = ChatCompletionChunkDelta(content=delta_msg.content)

        json_str = api_delta.model_dump_json()
        parsed = json.loads(json_str)

        assert parsed["content"] == "Final answer"
        assert "reasoning" not in parsed or parsed.get("reasoning") is None


class TestStreamingSimulation:
    """Simulate full streaming flow with reasoning parser."""

    @pytest.fixture
    def parser(self):
        parser = create_qwen3_parser()
        parser.reset_state()
        return parser

    def test_full_streaming_flow(self, parser):
        """Test complete streaming flow with reasoning."""
        encoder = StreamingJSONEncoder(
            response_id="chatcmpl-stream",
            model="qwen3-4b",
            object_type="chat.completion.chunk",
        )

        # Simulate streaming tokens
        stream_tokens = [
            ("Let me", [100, 101]),
            (" think", [102, 103]),
            (" step", [104, 105]),
            (" by step", [106, 107]),
            ("final<｜end▁of▁thinking▁｜>The", [108, 151649, 200]),
            (" answer", [201, 202]),
            (" is 42", [203, 204, 205]),
        ]

        chunks = []

        for delta_text, delta_ids in stream_tokens:
            delta_msg = parser.extract_streaming_with_tokens(
                delta_text=delta_text,
                delta_token_ids=delta_ids,
            )

            if delta_msg is None:
                continue  # Skip marker-only chunks

            # Build SSE chunk
            chunk = encoder.encode_chat_chunk(
                reasoning=delta_msg.reasoning,
                content=delta_msg.content,
            )
            chunks.append(chunk)

        # Verify we got chunks
        assert len(chunks) > 0

        # First chunks should have reasoning
        first_parsed = json.loads(chunks[0][6:])
        assert "reasoning" in first_parsed["choices"][0]["delta"]

        # Last chunks should have content only
        last_parsed = json.loads(chunks[-1][6:])
        assert "content" in last_parsed["choices"][0]["delta"]
        assert "reasoning" not in last_parsed["choices"][0]["delta"] or \
               last_parsed["choices"][0]["delta"].get("reasoning") is None


class TestOpenAIResponseFormatValidation:
    """Validate OpenAI response format compliance."""

    def test_reasoning_field_in_delta(self):
        """Test that reasoning field appears in delta as per design."""
        chunk = ChatCompletionChunk(
            id="chatcmpl-abc123",
            model="qwen3.5-122b",
            choices=[
                ChatCompletionChunkChoice(
                    delta=ChatCompletionChunkDelta(
                        reasoning="Let me think about this...",
                    ),
                )
            ],
        )

        json_str = chunk.model_dump_json()
        parsed = json.loads(json_str)

        # Validate OpenAI response format structure
        assert parsed["id"] == "chatcmpl-abc123"
        assert parsed["object"] == "chat.completion.chunk"
        assert parsed["model"] == "qwen3.5-122b"
        assert isinstance(parsed["choices"], list)
        assert len(parsed["choices"]) == 1

        choice = parsed["choices"][0]
        assert choice["index"] == 0
        assert "delta" in choice

        delta = choice["delta"]
        assert delta["reasoning"] == "Let me think about this..."
        assert delta.get("content") is None

    def test_reasoning_content_alias(self):
        """Test that reasoning_content alias is present for backwards compatibility."""
        msg = AssistantMessage(reasoning="Thinking...")

        json_str = msg.model_dump_json()
        parsed = json.loads(json_str)

        # Both reasoning and reasoning_content should be present
        assert parsed["reasoning"] == "Thinking..."
        assert parsed["reasoning_content"] == "Thinking..."