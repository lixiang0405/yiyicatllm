#!/bin/bash
# ============================================
# 环境搭建脚本
# 适用于: WSL2 + Conda + CUDA 12.8
# ============================================

set -e

ENV_NAME="yiyicat-llm"
PYTHON_VERSION="3.11"

echo "=========================================="
echo "  中科大智能问答助手 - 环境搭建"
echo "=========================================="

# Step 1: 创建 conda 环境
echo "[1/5] 创建 conda 环境: ${ENV_NAME}"
conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y
conda activate ${ENV_NAME}

# Step 2: 安装 PyTorch (CUDA 12.8)
echo "[2/5] 安装 PyTorch + CUDA 12.8"
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128

# Step 3: 安装项目依赖
echo "[3/5] 安装项目依赖"
pip install -r requirements.txt

# Step 4: 安装 LLaMA-Factory
echo "[4/5] 安装 LLaMA-Factory"
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
cd ..

# Step 5: 验证安装
echo "[5/5] 验证安装"
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"

echo ""
echo "=========================================="
echo "  环境搭建完成!"
echo "  激活环境: conda activate ${ENV_NAME}"
echo "=========================================="
