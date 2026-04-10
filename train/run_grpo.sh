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
#   - Actor: 策略模型 (DPO 合并后的模型)
#   - Reference: 参考模型 (DPO 合并后的模型，冻结)
#   - Reward: 奖励函数 (规则奖励，不需要 Reward Model)
#   - 不需要 Critic Model (这是 GRPO 的优势)

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VERL_CONFIG="${PROJECT_DIR}/train/verl_config.yaml"
DPO_MODEL_PATH="/root/autodl-tmp/ustc-qa-dpo-merged"
GRPO_DATA="${PROJECT_DIR}/data/grpo_prompts.parquet"

echo "=========================================="
echo "  中科大智能问答助手 - veRL GRPO 训练"
echo "=========================================="

# --- 检查 veRL 是否安装 ---
if ! python3 -c "import verl" 2>/dev/null; then
    echo "[ERROR] veRL 未安装，请先运行:"
    echo "  pip install verl vllm>=0.8.2"
    exit 1
fi

# --- 检查 DPO 合并模型 ---
if [ ! -d "${DPO_MODEL_PATH}" ]; then
    echo "[WARNING] DPO 合并模型不存在: ${DPO_MODEL_PATH}"
    # 回退到 SFT 合并模型
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

# --- 启动 GRPO 训练 ---
echo ""
echo "[3/3] 启动 veRL GRPO 训练..."

NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())")
echo "  可用 GPU: ${NUM_GPUS} 张"
echo "  Actor 模型: ${DPO_MODEL_PATH}"
echo "  奖励函数: 规则奖励 (reward_function.py)"
echo "  每个 prompt 采样: 4 个回答"

# 清理上一次的 Ray 进程，避免缓存导致配置不生效
ray stop --force 2>/dev/null || true
pkill -9 -f "vllm" 2>/dev/null || true
sleep 3

# vLLM v1 使用 CuMemAllocator 内存池，不兼容 expandable_segments
# 所以这里不能设置 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset PYTORCH_CUDA_ALLOC_CONF 2>/dev/null || true
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# ============================================
# veRL + vLLM 0.8+ (V1 engine) 配置
# 关键：需要 enforce_eager=False + free_cache_engine=True
# 参考：https://github.com/volcengine/verl/blob/main/docs/README_vllm0.8.md
# ============================================

python3 -m verl.trainer.main_ppo \
    --config-name="ppo_trainer" \
    data.train_files="${GRPO_DATA}" \
    data.val_files="${PROJECT_DIR}/data/grpo_eval.parquet" \
    data.prompt_key=prompt \
    data.max_prompt_length=256 \
    data.max_response_length=256 \
    data.train_batch_size=16 \
    data.trust_remote_code=true \
    actor_rollout_ref.model.path="${DPO_MODEL_PATH}" \
    actor_rollout_ref.model.trust_remote_code=true \
    '+actor_rollout_ref.model.override_config={attn_implementation: sdpa}' \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=64 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.002 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.use_torch_compile=false \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=20 \
    actor_rollout_ref.actor.optim.lr_scheduler_type=constant \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.enforce_eager=true \
    actor_rollout_ref.rollout.free_cache_engine=true \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    reward_model.enable=false \
    custom_reward_function.path=train/reward_function.py \
    custom_reward_function.name=compute_reward \
    algorithm.adv_estimator=grpo \
    trainer.total_epochs=1 \
    trainer.val_before_train=true \
    trainer.val_only=false \
    trainer.test_freq=50 \
    trainer.save_freq=100 \
    trainer.project_name=ustc-qa-grpo \
    trainer.n_gpus_per_node="${NUM_GPUS}" \
    trainer.experiment_name="grpo-$(date +%Y%m%d-%H%M%S)" \
    trainer.default_local_dir="${PROJECT_DIR}/outputs/ustc-qa-grpo" \
    'trainer.logger=["console"]'

echo ""
echo "=========================================="
echo "  GRPO 训练完成!"
echo "  模型保存至: outputs/ustc-qa-grpo"
echo ""
echo "  下一步:"
echo "    1. 评测模型: python train/evaluate_model.py"
echo "    2. 量化部署: python quantize/quantize_model.py"
echo "=========================================="
