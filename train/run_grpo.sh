#!/bin/bash
# ============================================
# GRPO 强化学习训练启动脚本 (自实现版本)
# 在 SFT + DPO 之后运行
# ============================================
#
# 全链路流程:
#   Step 1: SFT 微调 (run_train.sh)       → 学会领域知识和回答格式
#   Step 2: DPO 对齐 (run_dpo.sh)         → 学会生成高质量回答
#   Step 3: GRPO 强化学习 (本脚本)         → 进一步优化回答质量
#   Step 4: 量化部署 (quantize + serve)
#
# 实现方式:
#   自实现在线 GRPO，绕开 veRL hybrid engine 的 vLLM v1 兼容性问题
#   训练和推理交替进行，不同时占用 GPU 显存：
#     1. vLLM 离线批量生成 n 个回答 (占满 GPU)
#     2. 释放 vLLM 显存
#     3. PyTorch + PEFT LoRA 做 GRPO 策略更新 (占满 GPU)
#     4. 合并 LoRA 权重，下一轮用最新模型生成

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DPO_MODEL_PATH="/root/autodl-tmp/ustc-qa-dpo-merged"
GRPO_DATA="${PROJECT_DIR}/data/grpo_prompts.parquet"
OUTPUT_DIR="${PROJECT_DIR}/outputs/ustc-qa-grpo"

echo "=========================================="
echo "  中科大智能问答助手 - GRPO 训练"
echo "=========================================="

# --- 检查依赖 ---
python3 -c "import vllm, peft, torch" 2>/dev/null || {
    echo "[ERROR] 缺少依赖，请安装: pip install vllm peft torch transformers"
    exit 1
}

# --- 检查 DPO 合并模型 ---
if [ ! -d "${DPO_MODEL_PATH}" ]; then
    echo "[WARNING] DPO 合并模型不存在: ${DPO_MODEL_PATH}"
    DPO_MODEL_PATH="/root/autodl-tmp/ustc-qa-merged"
    if [ ! -d "${DPO_MODEL_PATH}" ]; then
        echo "[ERROR] SFT 合并模型也不存在，请先完成 SFT + DPO 训练流程"
        exit 1
    fi
    echo "  回退使用 SFT 合并模型: ${DPO_MODEL_PATH}"
fi

# --- 准备 GRPO 数据 ---
echo "[1/3] 准备 GRPO 训练数据..."
GRPO_EVAL_DATA="${PROJECT_DIR}/data/grpo_eval.parquet"
if [ ! -f "${GRPO_DATA}" ] || [ ! -f "${GRPO_EVAL_DATA}" ]; then
    python3 "${PROJECT_DIR}/train/prepare_grpo_data.py" \
        --sft-data "${PROJECT_DIR}/data/new_qa.json" \
        --pref-data "${PROJECT_DIR}/data/dpo_train_data.json" \
        --output "${GRPO_DATA}" \
        --eval-data "${PROJECT_DIR}/data/eval_data.json" \
        --eval-output "${GRPO_EVAL_DATA}"
fi
DATA_COUNT=$(python3 -c "import pandas as pd; print(len(pd.read_parquet('${GRPO_DATA}')))")
echo "  GRPO 数据: ${DATA_COUNT} 条 prompts"

# --- 测试奖励函数 ---
echo "[2/3] 测试奖励函数..."
python3 "${PROJECT_DIR}/train/reward_function.py"

# --- 清理残留进程 ---
echo ""
echo "[3/3] 启动 GRPO 训练..."
pkill -9 -f "vllm" 2>/dev/null || true
ray stop --force 2>/dev/null || true
sleep 2

# vLLM v1 CuMemAllocator 不兼容 expandable_segments
unset PYTORCH_CUDA_ALLOC_CONF 2>/dev/null || true

# --- 启动训练 ---
python3 "${PROJECT_DIR}/train/train_grpo.py" \
    --model "${DPO_MODEL_PATH}" \
    --data "${GRPO_DATA}" \
    --output "${OUTPUT_DIR}" \
    --num-samples 4 \
    --batch-size 16 \
    --epochs 1 \
    --lr 5e-6 \
    --lora-rank 32 \
    --lora-alpha 64 \
    --clip-epsilon 0.2 \
    --kl-coef 0.05 \
    --max-length 512 \
    --max-new-tokens 256 \
    --temperature 0.7 \
    --save-steps 50 \
    --skip-generate

echo ""
echo "=========================================="
echo "  GRPO 训练完成!"
echo "  模型保存至: ${OUTPUT_DIR}/final"
echo ""
echo "  下一步:"
echo "    1. 评测模型: python train/evaluate_model.py"
echo "    2. 量化部署: python quantize/quantize_model.py"
echo "=========================================="
