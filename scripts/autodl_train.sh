#!/bin/bash
# ============================================
# AutoDL 一键训练脚本
# 环境: 2×RTX 5090 32GB, PyTorch 2.8.0, CUDA 12.8
#
# 功能: 环境安装 → 模型下载 → 数据准备 → SFT训练 → 模型评测
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
echo "  [0/6] 验证 GPU 环境"
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
echo "  [1/6] 安装依赖"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# AutoDL 镜像已自带 PyTorch，只需安装其他依赖
# 使用清华源加速
pip install transformers>=4.46.0 datasets>=3.0.0 accelerate>=1.0.0 \
    peft>=0.13.0 trl>=0.12.0 deepspeed>=0.15.0 \
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
echo "  [2/6] 下载模型: ${MODEL_NAME}"
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
echo "  [3/6] 准备训练数据"
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
echo "  [4/6] 更新训练配置"
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
echo "    epochs: 6"
echo "    验证集: 10%"
# ==========================================
# Step 5: 开始 SFT LoRA 训练
# ==========================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [5/6] 开始 SFT LoRA 训练"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

bash train/run_train.sh

echo "  ✅ SFT 训练完成"

# ==========================================
# Step 6: 模型评测
# ==========================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [6/6] 模型评测 (基座 vs LoRA 微调)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python3 train/evaluate_model.py \
    --base-model "${MODEL_LOCAL_DIR}" \
    --lora-adapter outputs/ustc-qa-lora \
    --test-data data/new_qa.json \
    --num-samples ${NUM_EVAL_SAMPLES} \
    --output outputs/eval_report.json

echo "  ✅ 评测完成"

# ==========================================
# 最终汇总
# ==========================================
echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║              🎉 全部完成！                       ║"
echo "╠══════════════════════════════════════════════════╣"
echo "║                                                  ║"
echo "║  训练产物:                                       ║"
echo "║    LoRA 权重:  outputs/ustc-qa-lora/             ║"
echo "║    训练报告:   outputs/ustc-qa-lora/train_report.json  ║"
echo "║    GPU 监控:   outputs/ustc-qa-lora/gpu_stats.json     ║"
echo "║    评测报告:   outputs/eval_report.json          ║"
echo "║                                                  ║"
echo "║  下一步:                                         ║"
echo "║    1. 查看训练报告: cat outputs/ustc-qa-lora/train_report.json  ║"
echo "║    2. 合并权重: python3 train/merge_lora.py \\    ║"
echo "║         --base-model ${MODEL_LOCAL_DIR} \\        ║"
echo "║         --lora-adapter outputs/ustc-qa-lora      ║"
echo "║    3. 下载产物到本地后关机 (省钱!)               ║"
echo "║                                                  ║"
echo "╚══════════════════════════════════════════════════╝"
