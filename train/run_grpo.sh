#!/bin/bash
# ============================================
# veRL GRPO 强化学习训练启动脚本
# 在 SFT + DPO 之后运行
# ============================================
#
# 全链路流程:
#   Step 1: SFT 微调 (run_train.sh)       → 学会领域知识和回答格式
#   Step 2: DPO 对齐 (run_dpo.sh)         → 学会生成高质量回答
#   Step 3: GRPO 强化学习 (本脚本)         → 进一步优化回答质量
#   Step 4: 量化部署 (quantize + serve)
#
# GRPO vs PPO:
#   - PPO 需要 Critic Model 估计 baseline
#   - GRPO 用同一 prompt 的多个采样的平均 reward 作为 baseline
#   - GRPO 更省显存 (~25%)，训练更稳定
#
# veRL 架构:
#   - Actor: 策略模型 (SFT/DPO 后的模型)
#   - Reference: 参考模型 (SFT/DPO 后的模型，冻结)
#   - Reward: 奖励函数 (规则奖励，不需要 Reward Model)
#   - 不需要 Critic Model (这是 GRPO 的优势)

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VERL_CONFIG="${PROJECT_DIR}/train/verl_config.yaml"

echo "=========================================="
echo "  中科大智能问答助手 - veRL GRPO 训练"
echo "=========================================="

# --- 检查 veRL 是否安装 ---
if ! python3 -c "import verl" 2>/dev/null; then
    echo "[INFO] veRL 未安装，正在安装..."
    pip install verl
    echo "[INFO] veRL 安装完成"
fi

# --- 检查 SFT/DPO 模型 ---
SFT_MODEL_PATH="${PROJECT_DIR}/outputs/ustc-qa-merged"
if [ ! -d "${SFT_MODEL_PATH}" ]; then
    echo "[WARNING] 合并后的 SFT 模型不存在: ${SFT_MODEL_PATH}"
    echo "  请先运行:"
    echo "    1. bash train/run_train.sh      (SFT 微调)"
    echo "    2. bash train/run_dpo.sh        (DPO 对齐, 可选)"
    echo "    3. python train/merge_lora.py   (合并 LoRA 权重)"
    echo ""
    echo "  或者直接使用基座模型进行 GRPO 训练 (效果较差):"
    SFT_MODEL_PATH="Qwen/Qwen2.5-7B"
    echo "  使用模型: ${SFT_MODEL_PATH}"
fi

# --- 准备 GRPO 数据 ---
echo "[1/3] 准备 GRPO 训练数据..."
GRPO_DATA="${PROJECT_DIR}/data/grpo_prompts.json"
if [ ! -f "${GRPO_DATA}" ]; then
    python3 "${PROJECT_DIR}/train/prepare_grpo_data.py" \
        --sft-data "${PROJECT_DIR}/data/sample_data.json" \
        --output "${GRPO_DATA}"
fi
DATA_COUNT=$(python3 -c "import json; print(len(json.load(open('${GRPO_DATA}'))))")
echo "  GRPO 数据: ${DATA_COUNT} 条 prompts"

# --- 测试奖励函数 ---
echo "[2/3] 测试奖励函数..."
python3 "${PROJECT_DIR}/train/reward_function.py"

# --- 启动 GRPO 训练 ---
echo ""
echo "[3/3] 启动 veRL GRPO 训练..."

NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())")
echo "  可用 GPU: ${NUM_GPUS} 张"
echo "  Actor 模型: ${SFT_MODEL_PATH}"
echo "  奖励函数: 规则奖励 (reward_function.py)"

# veRL 使用 torchrun 启动
python3 -m verl.trainer.main_ppo \
    --config "${VERL_CONFIG}" \
    --actor_rollout_ref.model.path "${SFT_MODEL_PATH}" \
    --actor_rollout_ref.rollout.tensor_model_parallel_size 1 \
    --trainer.n_gpus_per_node "${NUM_GPUS}" \
    --trainer.project_name "ustc-qa-grpo" \
    --trainer.experiment_name "grpo-$(date +%Y%m%d-%H%M%S)" \
    --trainer.default_local_dir "${PROJECT_DIR}/outputs/ustc-qa-grpo"

echo ""
echo "=========================================="
echo "  GRPO 训练完成!"
echo "  模型保存至: outputs/ustc-qa-grpo"
echo ""
echo "  下一步: 量化部署"
echo "  python quantize/quantize_model.py \\"
echo "      --model-path outputs/ustc-qa-grpo \\"
echo "      --method awq --bits 4"
echo "=========================================="
