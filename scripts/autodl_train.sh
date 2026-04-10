#!/bin/bash
# ============================================
# AutoDL 一键训练脚本
# 环境: 2×RTX 5090 32GB, PyTorch 2.8.0, CUDA 12.8
#
# 功能: 环境安装 → 模型下载 → 数据准备 → SFT → DPO → GRPO 全链路训练
# 用法: bash scripts/autodl_train.sh
# ============================================

set -e

# ==========================================
# 配置区 (按需修改)
# ==========================================
MODEL_NAME="Qwen/Qwen2.5-7B"
MODEL_LOCAL_DIR="/root/autodl-tmp/Qwen2.5-7B"    # 模型下载到数据盘，关机不丢
PROJECT_DIR="/root/yiyicat-llm"                   # 项目目录
TRAIN_DATA="data/new_qa.json"                     # 训练数据文件名（去重清洗后）
NUM_EVAL_SAMPLES=200                              # 评测样本数

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║   中科大智能问答助手 - AutoDL 一键训练           ║"
echo "║   环境: 2×RTX 5090 32GB / CUDA 12.8             ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# ==========================================
# Step 0: 验证 GPU 环境
# ==========================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [0/11] 验证 GPU 环境"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python3 -c "
import torch
import sys

print(f'  Python:  {sys.version.split()[0]}')
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA:    {torch.version.cuda}')

if not torch.cuda.is_available():
    print('  ❌ CUDA 不可用！请检查镜像和驱动')
    sys.exit(1)

num_gpus = torch.cuda.device_count()
print(f'  GPU 数量: {num_gpus}')
for i in range(num_gpus):
    props = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB, sm_{props.major}.{props.minor})')

if num_gpus == 0:
    print('  ❌ 未检测到 GPU！')
    sys.exit(1)

print('  ✅ GPU 环境正常')
"

# ==========================================
# Step 1: 安装依赖
# ==========================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [1/11] 安装依赖"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# AutoDL 镜像已自带 PyTorch，只需安装其他依赖
# 使用清华源加速
pip install transformers>=4.46.0 datasets>=3.0.0 accelerate>=1.0.0 \
    peft>=0.13.0 trl>=0.12.0 deepspeed>=0.15.0 \
    pandas pyarrow \
    -i https://pypi.tuna.tsinghua.edu.cn/simple 2>&1 | tail -5

echo ""
echo "  安装 veRL + vLLM (GRPO 强化学习)..."
pip install verl vllm>=0.8.2 \
    -i https://pypi.tuna.tsinghua.edu.cn/simple 2>&1 | tail -5

echo ""
echo "  安装 LLaMA-Factory..."
if [ ! -d "${PROJECT_DIR}/LLaMA-Factory" ]; then
    cd "${PROJECT_DIR}"
    git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
    cd LLaMA-Factory
    pip install -e ".[torch,metrics]" -i https://pypi.tuna.tsinghua.edu.cn/simple 2>&1 | tail -5
    cd "${PROJECT_DIR}"
else
    echo "  LLaMA-Factory 已存在，跳过安装"
fi

echo "  ✅ 依赖安装完成"

# ==========================================
# Step 2: 下载 Qwen2.5-7B 模型
# ==========================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [2/11] 下载模型: ${MODEL_NAME}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ -d "${MODEL_LOCAL_DIR}" ] && [ -f "${MODEL_LOCAL_DIR}/config.json" ]; then
    echo "  模型已存在: ${MODEL_LOCAL_DIR}，跳过下载"
else
    echo "  下载到数据盘: ${MODEL_LOCAL_DIR}"
    echo "  (约 14GB，预计 5-10 分钟)"

    # 优先使用 modelscope (国内更快)
    pip install modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple 2>&1 | tail -3

    python3 -c "
from modelscope import snapshot_download
snapshot_download('Qwen/Qwen2.5-7B', local_dir='${MODEL_LOCAL_DIR}')
print('  ✅ 模型下载完成')
"
fi

echo "  模型路径: ${MODEL_LOCAL_DIR}"

# ==========================================
# Step 3: 准备训练数据
# ==========================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [3/11] 准备训练数据"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

cd "${PROJECT_DIR}"

# 检查训练数据 new_qa.json
if [ -f "${TRAIN_DATA}" ]; then
    DATA_COUNT=$(python3 -c "import json; print(len(json.load(open('${TRAIN_DATA}'))))")
    echo "  训练数据: ${DATA_COUNT} 条 (来自 ${TRAIN_DATA})"
else
    echo "  [WARN] ${TRAIN_DATA} 不存在，使用 sample_data.json"
    cp sample_data.json "${TRAIN_DATA}"
    DATA_COUNT=$(python3 -c "import json; print(len(json.load(open('${TRAIN_DATA}'))))")
    echo "  训练数据: ${DATA_COUNT} 条 (示例数据)"
fi

echo "  ✅ 数据准备完成"

# ==========================================
# Step 4: 修改配置中的模型路径
# ==========================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [4/11] 更新训练配置"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 将 yaml 中的模型路径替换为本地路径
sed -i "s|model_name_or_path:.*|model_name_or_path: ${MODEL_LOCAL_DIR}|" train/train_lora.yaml
echo "  模型路径已更新: ${MODEL_LOCAL_DIR}"

# 检测 GPU 数量，如果单卡则去掉 DeepSpeed
NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())")
if [ "${NUM_GPUS}" -eq 1 ]; then
    echo "  单卡模式: 移除 DeepSpeed 配置"
    sed -i '/^deepspeed:/d' train/train_lora.yaml
fi
    echo "    batch_size/卡: 4"
echo "    grad_accum: 4"
echo "    有效 batch_size: $((4 * 4 * NUM_GPUS))"
echo "    epochs: 10"
echo "    验证集: 10%"
# ==========================================
# Step 5: 开始 SFT LoRA 训练
# ==========================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [5/11] 开始 SFT LoRA 训练"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

bash train/run_train.sh

echo "  ✅ SFT 训练完成"

# ==========================================
# Step 6: 合并 SFT LoRA 权重
# ==========================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [6/11] 合并 SFT LoRA 权重"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python3 train/merge_lora.py \
    --base-model "${MODEL_LOCAL_DIR}" \
    --lora-adapter outputs/ustc-qa-lora \
    --output outputs/ustc-qa-merged

echo "  ✅ SFT LoRA 合并完成: outputs/ustc-qa-merged"

# ==========================================
# Step 7: 生成 DPO rejected 回答
# ==========================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [7/11] 生成 DPO rejected 回答"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python3 train/generate_rejected.py \
    --model outputs/ustc-qa-merged \
    --data data/preference_data.json \
    --output data/preference_data.json

echo "  ✅ rejected 回答生成完成"

# ==========================================
# Step 8: DPO 偏好对齐训练
# ==========================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [8/11] DPO 偏好对齐训练"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

bash train/run_dpo.sh

echo "  ✅ DPO 训练完成"

# ==========================================
# Step 9: 合并 DPO LoRA 权重
# ==========================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [9/11] 合并 DPO LoRA 权重"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python3 train/merge_lora.py \
    --base-model outputs/ustc-qa-merged \
    --lora-adapter outputs/ustc-qa-dpo \
    --output outputs/ustc-qa-dpo-merged

echo "  ✅ DPO LoRA 合并完成: outputs/ustc-qa-dpo-merged"

# ==========================================
# Step 10: GRPO 强化学习训练
# ==========================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [10/11] GRPO 强化学习训练"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

bash train/run_grpo.sh

echo "  ✅ GRPO 训练完成"

# ==========================================
# Step 11: 最终评测
# ==========================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [11/11] 最终模型评测"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python3 train/evaluate_model.py \
    --base-model "${MODEL_LOCAL_DIR}" \
    --test-data data/preference_data.json \
    --num-samples ${NUM_EVAL_SAMPLES} \
    --output outputs/eval_report.json

echo "  ✅ 评测完成"

# ==========================================
# 最终汇总
# ==========================================
echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║              🎉 全链路训练完成！                 ║"
echo "╠══════════════════════════════════════════════════╣"
echo "║                                                  ║"
echo "║  训练产物:                                       ║"
echo "║    SFT LoRA:     outputs/ustc-qa-lora/           ║"
echo "║    SFT 合并:     outputs/ustc-qa-merged/         ║"
echo "║    DPO LoRA:     outputs/ustc-qa-dpo/            ║"
echo "║    DPO 合并:     outputs/ustc-qa-dpo-merged/     ║"
echo "║    GRPO:         outputs/ustc-qa-grpo/           ║"
echo "║    评测报告:     outputs/eval_report.json        ║"
echo "║                                                  ║"
echo "║  下载产物到本地后关机 (省钱!)                    ║"
echo "║                                                  ║"
echo "╚══════════════════════════════════════════════════╝"
