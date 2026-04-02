# SPDX-License-Identifier: Apache-2.0
"""
Reasoning output parsers for models with thinking/reasoning capabilities.

This module provides parsers for extracting reasoning content from models
that use special markers to separate reasoning from final responses.

## Two Parser Architectures

### 1. BaseThinkingReasoningParser (text-based, simple scenarios)

Located in `think_parser.py`. This parser family uses text-based marker detection
and is suitable for models where markers are single, distinct text strings.

**Use when**:
- Markers are simple text tags (e.g., `{{--REASONING--}}`...`{{/REASONING--}}`)
- Token ID detection is not needed or not available
- Streaming with incremental text chunks

**Parser classes**:
- `BaseThinkingReasoningParser`: Abstract base with full streaming support
- `QwenThinkParser`: Qwen3/Qwen3.5 unified parser (uses `...`)
- `DeepSeekR1ReasoningParser`: DeepSeek-R1 parser (uses similar markers)

**Limitations**:
- Cannot detect markers that span multiple tokens
- May leak partial markers in streaming if delta contains partial tag

### 2. ReasoningOutputParser (token ID support, complex scenarios)

Located in `output_parser.py`. This parser extends the base with token ID awareness
for precise marker detection during streaming.

**Use when**:
- Markers may span multiple tokens
- Token IDs are available and stable
- Need precise marker detection without leakage

**Parser classes**:
- `ReasoningOutputParser`: Token-aware parser with state tracking

**Factory functions**:
- `create_qwen3_parser()`: Creates configured parser (but see WARNING below)
- `create_qwen35_parser()`: Creates configured parser (but see WARNING below)
- `create_deepseek_r1_parser()`: Creates configured parser

**IMPORTANT WARNING for Qwen3/Qwen3.5**:
The token IDs 151648/151649 in Qwen tokenizer are NOT thinking markers!
They are Russian/Thai characters. Use `QwenThinkParser` (text-based) instead,
or use `ReasoningOutputParser` with token_ids=None.

### When to Use Which

| Model | Recommended Parser | Reason |
|-------|-------------------|--------|
| Qwen3/Qwen3.5 | `QwenThinkParser` (via registry) | Token IDs unreliable, text-based works |
| DeepSeek-R1 | `DeepSeekR1ReasoningParser` | Similar markers, text-based sufficient |
| Future models with multi-token markers | `ReasoningOutputParser` | Token ID detection prevents leakage |

## Usage

```python
from vllm_mlx.reasoning import get_parser, list_parsers

# List available parsers
print(list_parsers())  # ['qwen3', 'qwen3.5', 'qwen35', 'deepseek_r1', ...]

# Get a parser by name (returns class, not instance)
parser_class = get_parser("qwen3")
parser = parser_class()

# Extract reasoning from complete output
reasoning, content = parser.extract_reasoning(model_output)

# For streaming
parser.reset_state()
for delta in stream:
    msg = parser.extract_reasoning_streaming(prev, curr, delta)
    if msg:
        # msg.reasoning and/or msg.content will be populated
        ...
```

## Registry

The parser registry allows registering custom parsers:

```python
from vllm_mlx.reasoning import register_parser

class MyCustomParser(BaseThinkingReasoningParser):
    @property
    def start_token(self) -> str:
        return "<reasoning>"
    @property
    def end_token(self) -> str:
        return "</reasoning>"

register_parser("my_model", MyCustomParser)
```
"""

from .base import DeltaMessage, ReasoningParser
from .think_parser import BaseThinkingReasoningParser

# Parser registry
_REASONING_PARSERS: dict[str, type[ReasoningParser]] = {}


def register_parser(name: str, parser_class: type[ReasoningParser]) -> None:
    """
    Register a reasoning parser.

    Args:
        name: Name to register the parser under (e.g., "qwen3").
        parser_class: The parser class to register.
    """
    _REASONING_PARSERS[name] = parser_class


def get_parser(name: str) -> type[ReasoningParser]:
    """
    Get a reasoning parser class by name.

    Args:
        name: Name of the parser (e.g., "qwen3", "deepseek_r1").

    Returns:
        The parser class (not an instance).

    Raises:
        KeyError: If parser name is not found.
    """
    if name not in _REASONING_PARSERS:
        available = list(_REASONING_PARSERS.keys())
        raise KeyError(
            f"Reasoning parser '{name}' not found. Available parsers: {available}"
        )
    return _REASONING_PARSERS[name]


def list_parsers() -> list[str]:
    """
    List available parser names.

    Returns:
        List of registered parser names.
    """
    return list(_REASONING_PARSERS.keys())


def _register_builtin_parsers():
    """Register built-in parsers."""
    from .deepseek_r1_parser import DeepSeekR1ReasoningParser
    from .gpt_oss_parser import GptOssReasoningParser
    from .harmony_parser import HarmonyReasoningParser
    from .qwen_think_parser import QwenThinkParser

    # Qwen3/Qwen3.5 share the same parser (identical markers)
    register_parser("qwen3", QwenThinkParser)
    register_parser("qwen3.5", QwenThinkParser)
    register_parser("qwen35", QwenThinkParser)  # Alias for qwen3.5
    register_parser("deepseek_r1", DeepSeekR1ReasoningParser)
    register_parser("gpt_oss", GptOssReasoningParser)
    register_parser("harmony", HarmonyReasoningParser)


# Register built-in parsers on module load
_register_builtin_parsers()


# Import thinking budget functionality
from .thinking_budget import (
    ThinkingBudgetCriteria,
    create_thinking_budget_criteria,
)


__all__ = [
    # Base classes
    "ReasoningParser",
    "DeltaMessage",
    "BaseThinkingReasoningParser",
    "ReasoningOutputParser",
    # Parser classes
    "QwenThinkParser",
    # Factory functions
    "create_qwen3_parser",
    "create_qwen35_parser",
    "create_deepseek_r1_parser",
    # Marker configs
    "QWEN3_MARKERS",
    "QWEN35_MARKERS",
    "DEEPSEEK_R1_MARKERS",
    # Registry functions
    "register_parser",
    "get_parser",
    "list_parsers",
    # Thinking budget
    "ThinkingBudgetCriteria",
    "create_thinking_budget_criteria",
]