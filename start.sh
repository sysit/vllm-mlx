#!/bin/bash
# vllm-mlx 启动脚本 - Qwen3.5-122B-A10B (MLLM)

cd ~/vllm-mlx
source venv/bin/activate

MODEL=~/models/mlx-community/Qwen3.5-122B-A10B-4bit

python -m vllm_mlx.server \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port 8000 \
    --mllm \
    --reasoning-parser qwen35