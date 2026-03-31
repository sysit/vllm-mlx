#!/usr/bin/env python3
"""
Quick verification script for ReasoningOutputParser.

Usage:
    python scripts/verify_output_parser.py
"""

from vllm_mlx.reasoning import (
    ReasoningOutputParser,
    create_qwen3_parser,
    DeltaMessage,
)


def test_non_streaming():
    """Test non-streaming extraction."""
    print("\n=== Non-Streaming Extraction ===\n")
    
    parser = create_qwen3_parser()
    
    # Example output with both markers
    output = "<｜begin▁of▁thinking▁｜>Let me think step by step... First, analyze the problem. Second, compute the result.<｜end▁of▁thinking▁｜>The answer is 42."
    
    reasoning, content = parser.extract_reasoning(output)
    
    print(f"Input: {output[:50]}...")
    print(f"\nReasoning: {reasoning[:50]}...")
    print(f"Content: {content}")
    
    assert reasoning == "Let me think step by step... First, analyze the problem. Second, compute the result."
    assert content == "The answer is 42."
    print("\n✅ Non-streaming extraction works correctly")


def test_token_based_streaming():
    """Test token-based streaming extraction."""
    print("\n=== Token-Based Streaming Extraction ===\n")
    
    parser = create_qwen3_parser()
    parser.reset_state()
    
    # Simulate streaming output with token IDs
    # Note: 151648 = start_token_id, 151649 = end_token_id
    
    streaming_chunks = [
        ("Let me think", [100, 101, 102]),
        (" step by step", [103, 104, 105]),
        ("... analyzing<｜end▁of▁thinking▁｜>The answer", [106, 151649, 200]),
        (" is 42.", [201, 202]),
    ]
    
    reasoning_parts = []
    content_parts = []
    
    for delta_text, delta_token_ids in streaming_chunks:
        msg = parser.extract_streaming_with_tokens(delta_text, delta_token_ids)
        if msg:
            if msg.reasoning:
                reasoning_parts.append(msg.reasoning)
            if msg.content:
                content_parts.append(msg.content)
            print(f"Delta: '{delta_text[:30]}...' → reasoning={bool(msg.reasoning)}, content={bool(msg.content)}")
    
    full_reasoning = "".join(reasoning_parts)
    full_content = "".join(content_parts)
    
    print(f"\nFull reasoning: {full_reasoning}")
    print(f"Full content: {full_content}")
    
    assert "Let me think" in full_reasoning
    assert "The answer is 42." in full_content
    print("\n✅ Token-based streaming extraction works correctly")


def test_state_tracking():
    """Test state tracking across chunks."""
    print("\n=== State Tracking ===\n")
    
    parser = create_qwen3_parser()
    parser.reset_state()
    
    # Chunk 1: Reasoning phase
    msg1 = parser.extract_streaming_with_tokens("thinking", [100])
    print(f"After chunk 1: in_content_phase={parser._in_content_phase}")
    print(f"previous_token_ids={parser._previous_token_ids}")
    
    # Chunk 2: End marker appears
    msg2 = parser.extract_streaming_with_tokens("<｜end▁of▁thinking▁｜>content", [151649, 200])
    print(f"After chunk 2: in_content_phase={parser._in_content_phase}")
    
    # Chunk 3: Pure content
    msg3 = parser.extract_streaming_with_tokens("more content", [201, 202])
    print(f"After chunk 3: msg.reasoning={msg3.reasoning}, msg.content={msg3.content}")
    
    assert parser._in_content_phase == True
    assert msg3.content == "more content"
    print("\n✅ State tracking works correctly")


def main():
    print("=" * 60)
    print("ReasoningOutputParser Verification Script")
    print("=" * 60)
    
    test_non_streaming()
    test_token_based_streaming()
    test_state_tracking()
    
    print("\n" + "=" * 60)
    print("All verifications passed! ✅")
    print("=" * 60)


if __name__ == "__main__":
    main()