#!/bin/bash
# ============================================
# DPO 偏好对齐训练启动脚本
# 在 SFT 微调之后运行
# ============================================
#
# 全链路流程:
#   Step 1: SFT 微调 (run_train.sh)     → 学会领域知识和回答格式
#   Step 2: DPO 对齐 (本脚本)            → 学会生成高质量、详细的回答
#   Step 3: 合并权重 (merge_lora.py)     → 合并 SFT + DPO 的 LoRA 权重
#   Step 4: 量化部署 (quantize + serve)

set -e

TRAIN_CONFIG="train/train_dpo.yaml"
LLAMA_FACTORY_DIR="LLaMA-Factory"
PREF_DATA_FILE="$(pwd)/data/preference_data.json"

echo "=========================================="
echo "  中科大智能问答助手 - DPO 偏好对齐训练"
echo "=========================================="

# --- 检查 SFT 模型是否存在 ---
if [ ! -d "outputs/ustc-qa-lora" ]; then
    echo "[ERROR] SFT 模型不存在，请先运行 bash train/run_train.sh"
    echo "  DPO 需要在 SFT 微调后的模型基础上进行"
    exit 1
fi

# --- 检查 LLaMA-Factory ---
if [ ! -d "${LLAMA_FACTORY_DIR}" ]; then
    echo "[ERROR] LLaMA-Factory 未安装，请先运行 scripts/setup_env.sh"
    exit 1
fi

# --- 注册偏好数据集 ---
echo "[1/3] 注册偏好数据集..."
DATASET_INFO="${LLAMA_FACTORY_DIR}/data/dataset_info.json"

python3 -c "
import json
with open('${DATASET_INFO}', 'r') as f:
    info = json.load(f)
info['ustc_qa_preference'] = {
    'file_name': '${PREF_DATA_FILE}',
    'ranking': True,
    'columns': {
        'prompt': 'instruction',
        'query': 'input',
        'chosen': 'chosen',
        'rejected': 'rejected'
    }
}
with open('${DATASET_INFO}', 'w') as f:
    json.dump(info, f, indent=2, ensure_ascii=False)
print('  偏好数据集注册成功: ustc_qa_preference')
"

# --- 检查偏好数据 ---
echo "[2/3] 检查偏好数据..."
if [ ! -f "${PREF_DATA_FILE}" ]; then
    echo "[ERROR] 偏好数据不存在: ${PREF_DATA_FILE}"
    exit 1
fi

DATA_COUNT=$(python3 -c "import json; print(len(json.load(open('${PREF_DATA_FILE}'))))")
echo "  偏好数据: ${DATA_COUNT} 条"

# --- 开始 DPO 训练 ---
echo "[3/3] 开始 DPO 训练..."

NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())")
echo "  可用 GPU: ${NUM_GPUS} 张"

if [ "${NUM_GPUS}" -gt 1 ]; then
    echo "  模式: 多卡分布式训练"
    FORCE_TORCHRUN=1 NNODES=1 NPROC_PER_NODE=${NUM_GPUS} \
        llamafactory-cli train ${TRAIN_CONFIG}
else
    echo "  模式: 单卡训练"
    llamafactory-cli train ${TRAIN_CONFIG}
fi

echo ""
echo "=========================================="
echo "  DPO 训练完成!"
echo "  DPO LoRA 权重保存至: outputs/ustc-qa-dpo"
echo ""
echo "  下一步: 合并 LoRA 权重"
echo "  python train/merge_lora.py \\"
echo "      --base-model Qwen/Qwen2.5-7B \\"
echo "      --lora-adapter outputs/ustc-qa-dpo \\"
echo "      --output outputs/ustc-qa-merged"
echo "=========================================="
