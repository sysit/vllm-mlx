# SPDX-License-Identifier: Apache-2.0
"""
Model Type Detector for vllm-mlx.

方案B简化版：纯名称模式匹配，避免 AutoConfig 网络加载问题。

检测逻辑：
- VL 模型：名称包含 VL 相关关键词
- Audio 模型：名称包含 Audio 相关关键词
- Text 模型：默认（无 VL/Audio 特征）
"""
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# VL 模型常见名称模式
VL_PATTERNS = [
    "vl", "vision", "llava", "paligemma", "pixtral",
    "idefics", "florence", "internvl", "qwen-vl", "qwen2-vl", "qwen3-vl",
    "molmo", "phi3-v", "gemma3", "gemma-3", "medgemma",
    "qwen3.5-122", "qwen3.5-35",  # Qwen3.5 MoE VL models (have vision_config)
    "qwen3_5moe",  # MLX naming variant
]

# Audio 模型常见名称模式
AUDIO_PATTERNS = ["whisper", "asr", "parakeet", "audio"]


def detect_model_type(model_path: str | Path) -> dict[str, Any]:
    """
    使用名称模式检测模型类型。
    
    Args:
        model_path: 模型路径（本地目录或 HuggingFace 名称）
    
    Returns:
        {
            "is_vl": bool,
            "is_audio": bool,
            "arch": str,
            "model_type": str,
        }
    """
    model_str = str(model_path).lower()
    
    # VL 检测
    is_vl = any(p in model_str for p in VL_PATTERNS)
    
    # Audio 检测
    is_audio = any(p in model_str for p in AUDIO_PATTERNS)
    
    return {
        "is_vl": is_vl,
        "is_audio": is_audio,
        "arch": "unknown",
        "model_type": "",
    }


def is_mllm_model(model_name: str) -> bool:
    """
    Check if model is a multimodal language model (MLLM/VLM).
    
    使用名称模式匹配，零维护。
    """
    result = detect_model_type(model_name)
    return result["is_vl"]


# Backwards compatibility
is_vlm_model = is_mllm_model