#!/bin/bash
# ============================================
# vLLM 推理服务启动脚本
# 适配 RTX 5070 Laptop 8GB 显存
# ============================================

set -e

# --- 默认配置 ---
MODEL_PATH="${1:-outputs/ustc-qa-quantized-awq-int4}"
PORT="${2:-8000}"
MAX_MODEL_LEN=2048
GPU_MEMORY_UTILIZATION=0.90

echo "=========================================="
echo "  中科大智能问答助手 - 推理服务"
echo "=========================================="
echo "  模型: ${MODEL_PATH}"
echo "  端口: ${PORT}"
echo "  最大序列长度: ${MAX_MODEL_LEN}"
echo "  GPU 显存利用率: ${GPU_MEMORY_UTILIZATION}"
echo "=========================================="

# 检查模型路径
if [ ! -d "${MODEL_PATH}" ]; then
    echo "[ERROR] 模型路径不存在: ${MODEL_PATH}"
    echo "  请先完成训练、合并和量化步骤"
    exit 1
fi

# 检测量化方式
QUANT_FLAG=""
if echo "${MODEL_PATH}" | grep -qi "gptq"; then
    QUANT_FLAG="--quantization gptq"
    echo "  量化方式: GPTQ"
elif echo "${MODEL_PATH}" | grep -qi "awq"; then
    QUANT_FLAG="--quantization awq"
    echo "  量化方式: AWQ"
else
    echo "  量化方式: 无 (FP16/BF16)"
fi

echo ""
echo "启动 vLLM 服务..."
echo "API 地址: http://localhost:${PORT}/v1"
echo "文档地址: http://localhost:${PORT}/docs"
echo ""

# 启动 vLLM OpenAI-compatible server
python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --port ${PORT} \
    --max-model-len ${MAX_MODEL_LEN} \
    --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
    --enable-prefix-caching \
    --trust-remote-code \
    ${QUANT_FLAG}
