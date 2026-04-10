#!/bin/bash
# ============================================
# 训练启动脚本
# 支持单卡和多卡训练
# 自动记录: 训练时间、GPU显存/利用率、训练报告
# ============================================

set -e

# --- 配置 ---
TRAIN_CONFIG="train/train_lora.yaml"
LLAMA_FACTORY_DIR="LLaMA-Factory"
DATA_FILE="$(pwd)/data/new_qa.json"
PREF_DATA_FILE="$(pwd)/data/preference_data.json"
EVAL_DATA_FILE="$(pwd)/data/eval_data.json"
DPO_TRAIN_DATA_FILE="$(pwd)/data/dpo_train_data.json"
OUTPUT_DIR="outputs/ustc-qa-lora"
GPU_STATS_FILE="${OUTPUT_DIR}/gpu_stats.json"
TRAIN_REPORT_FILE="${OUTPUT_DIR}/train_report.json"

echo "=========================================="
echo "  中科大智能问答助手 - LoRA 微调训练"
echo "=========================================="

# --- Step 0: 检查 LLaMA-Factory 是否安装 ---
if [ ! -d "${LLAMA_FACTORY_DIR}" ]; then
    echo "[ERROR] LLaMA-Factory 未安装，请先运行 scripts/setup_env.sh"
    exit 1
fi

# --- Step 0.5: 从偏好数据中切分公共验证集 ---
echo "[0/6] 从偏好数据中切分公共验证集..."
python3 -c "
import json, random

random.seed(42)

# 从偏好数据中切分 200~300 条作为公共验证集（SFT + DPO 共用）
with open('${PREF_DATA_FILE}', 'r') as f:
    pref_data = json.load(f)

total = len(pref_data)
eval_size = min(300, max(200, total // 10))  # 200~300 条，或总量的 10%

random.shuffle(pref_data)
eval_data = pref_data[:eval_size]
dpo_train_data = pref_data[eval_size:]

# 保存公共验证集（SFT 阶段用 output 做验证，DPO 阶段 generate_rejected 后用 chosen/rejected 做验证）
with open('${EVAL_DATA_FILE}', 'w') as f:
    json.dump(eval_data, f, indent=2, ensure_ascii=False)

# 保存 DPO 训练数据（去掉验证集部分）
with open('${DPO_TRAIN_DATA_FILE}', 'w') as f:
    json.dump(dpo_train_data, f, indent=2, ensure_ascii=False)

print(f'  偏好数据: {total} 条 → DPO 训练 {len(dpo_train_data)} 条 + 公共验证 {eval_size} 条')
print(f'  公共验证集: ${EVAL_DATA_FILE}')
print(f'  DPO 训练集: ${DPO_TRAIN_DATA_FILE}')
"

# --- Step 1: 注册数据集到 LLaMA-Factory ---
echo "[1/6] 注册数据集..."
DATASET_INFO="$(pwd)/${LLAMA_FACTORY_DIR}/data/dataset_info.json"
DATASET_DIR_ABS="$(pwd)/${LLAMA_FACTORY_DIR}/data"

python3 -c "
import json
with open('${DATASET_INFO}', 'r') as f:
    info = json.load(f)
info['ustc_qa'] = {
    'file_name': '${DATA_FILE}',
    'columns': {
        'prompt': 'instruction',
        'query': 'input',
        'response': 'output'
    }
}
info['ustc_qa_eval'] = {
    'file_name': '${EVAL_DATA_FILE}',
    'columns': {
        'prompt': 'instruction',
        'query': 'input',
        'response': 'output'
    }
}
with open('${DATASET_INFO}', 'w') as f:
    json.dump(info, f, indent=2, ensure_ascii=False)
print('  数据集注册成功: ustc_qa -> ${DATA_FILE}')
"

# 将 yaml 中的 dataset_dir 替换为绝对路径
sed -i "s|^dataset_dir:.*|dataset_dir: ${DATASET_DIR_ABS}|" ${TRAIN_CONFIG}
echo "  dataset_dir 已更新: ${DATASET_DIR_ABS}"

# --- Step 2: 检查训练数据 ---
echo "[2/6] 检查训练数据..."
if [ ! -f "${DATA_FILE}" ]; then
    echo "  [WARN] new_qa.json 不存在，使用 sample_data.json"
    cp data/sample_data.json data/new_qa.json
    DATA_FILE="$(pwd)/data/new_qa.json"
fi

DATA_COUNT=$(python3 -c "import json; print(len(json.load(open('${DATA_FILE}'))))")
echo "  训练数据: ${DATA_COUNT} 条"

# --- Step 3: 采集环境信息 ---
echo "[3/6] 采集环境信息..."
mkdir -p "${OUTPUT_DIR}"

NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())")
echo "  可用 GPU: ${NUM_GPUS} 张"

# 采集 GPU 和环境信息，保存到训练报告
python3 -c "
import json, torch, platform, subprocess

env_info = {
    'pytorch_version': torch.__version__,
    'cuda_version': torch.version.cuda,
    'python_version': platform.python_version(),
    'os': platform.platform(),
    'num_gpus': torch.cuda.device_count(),
    'gpus': [],
}
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    env_info['gpus'].append({
        'index': i,
        'name': props.name,
        'total_memoryory_gb': round(props.total_memory / 1024**3, 1),
        'compute_capability': f'{props.major}.{props.minor}',
    })

# 保存环境信息
with open('${TRAIN_REPORT_FILE}', 'w') as f:
    json.dump({'environment': env_info}, f, indent=2, ensure_ascii=False)

for gpu in env_info['gpus']:
    print(f\"  GPU {gpu['index']}: {gpu['name']} ({gpu['total_memoryory_gb']} GB, sm_{gpu['compute_capability']})\")
"

# --- Step 4: 启动 GPU 监控 (后台) ---
echo "[4/6] 启动 GPU 监控..."
python3 train/gpu_monitor.py --output "${GPU_STATS_FILE}" --interval 5 &
GPU_MONITOR_PID=$!
echo "  GPU 监控已启动 (PID=${GPU_MONITOR_PID})"

# 确保脚本退出时停止 GPU 监控
cleanup() {
    echo ""
    echo "  停止 GPU 监控..."
    if kill -0 ${GPU_MONITOR_PID} 2>/dev/null; then
        kill ${GPU_MONITOR_PID}
        wait ${GPU_MONITOR_PID} 2>/dev/null || true
    fi
}
trap cleanup EXIT

# --- Step 5: 开始训练 ---
echo "[5/6] 开始训练..."

TRAIN_START_TIME=$(date +%s)
TRAIN_START_TIME_STR=$(date '+%Y-%m-%d %H:%M:%S')
echo "  训练开始时间: ${TRAIN_START_TIME_STR}"

if [ "${NUM_GPUS}" -gt 1 ]; then
    echo "  模式: 多卡分布式训练 (DeepSpeed ZeRO-2)"
    echo ""
    FORCE_TORCHRUN=1 NNODES=1 NPROC_PER_NODE=${NUM_GPUS} \
        llamafactory-cli train ${TRAIN_CONFIG}
else
    echo "  模式: 单卡训练"
    echo ""
    llamafactory-cli train ${TRAIN_CONFIG}
fi

TRAIN_END_TIME=$(date +%s)
TRAIN_END_TIME_STR=$(date '+%Y-%m-%d %H:%M:%S')
TRAIN_DURATION=$((TRAIN_END_TIME - TRAIN_START_TIME))
TRAIN_MINUTES=$((TRAIN_DURATION / 60))
TRAIN_SECONDS=$((TRAIN_DURATION % 60))

echo ""
echo "  训练结束时间: ${TRAIN_END_TIME_STR}"
echo "  训练总耗时: ${TRAIN_MINUTES}分${TRAIN_SECONDS}秒"

# --- 停止 GPU 监控 ---
echo ""
echo "  停止 GPU 监控..."
if kill -0 ${GPU_MONITOR_PID} 2>/dev/null; then
    kill ${GPU_MONITOR_PID}
    wait ${GPU_MONITOR_PID} 2>/dev/null || true
fi
# 取消 trap，避免重复执行
trap - EXIT

# --- 生成训练报告 ---
echo ""
echo "  生成训练报告..."

python3 -c "
import json
from pathlib import Path

# 读取已有的环境信息
report_path = '${TRAIN_REPORT_FILE}'
with open(report_path, 'r') as f:
    report = json.load(f)

# 添加训练时间信息
report['training_time'] = {
    'start_time': '${TRAIN_START_TIME_STR}',
    'end_time': '${TRAIN_END_TIME_STR}',
    'duration_seconds': ${TRAIN_DURATION},
    'duration_human': '${TRAIN_MINUTES}分${TRAIN_SECONDS}秒',
}

# 添加训练配置
report['training_config'] = {
    'data_file': '${DATA_FILE}',
    'data_count': ${DATA_COUNT},
    'config_file': '${TRAIN_CONFIG}',
    'num_gpus': ${NUM_GPUS},
    'distributed': ${NUM_GPUS} > 1,
}

# 读取 GPU 监控数据的汇总
gpu_stats_path = '${GPU_STATS_FILE}'
if Path(gpu_stats_path).exists():
    with open(gpu_stats_path, 'r') as f:
        gpu_data = json.load(f)
    report['gpu_stats_summary'] = gpu_data.get('summary', {})

# 读取 LLaMA-Factory 的 trainer_state (包含 loss 曲线)
trainer_state_path = Path('${OUTPUT_DIR}') / 'trainer_state.json'
if trainer_state_path.exists():
    with open(trainer_state_path, 'r') as f:
        state = json.load(f)
    log_history = state.get('log_history', [])

    # 提取 train loss 和 eval loss
    train_losses = [h for h in log_history if 'loss' in h and 'eval_loss' not in h]
    eval_losses = [h for h in log_history if 'eval_loss' in h]

    report['loss_curve'] = {
        'train_loss_start': train_losses[0]['loss'] if train_losses else None,
        'train_loss_end': train_losses[-1]['loss'] if train_losses else None,
        'eval_loss_start': eval_losses[0]['eval_loss'] if eval_losses else None,
        'eval_loss_end': eval_losses[-1]['eval_loss'] if eval_losses else None,
        'eval_loss_best': min((h['eval_loss'] for h in eval_losses), default=None),
        'num_train_log_entries': len(train_losses),
        'num_eval_log_entries': len(eval_losses),
    }

# 保存完整报告
with open(report_path, 'w') as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

# 打印报告摘要
print('')
print('=' * 60)
print('  训练报告摘要')
print('=' * 60)
print(f\"  训练数据: {report['training_config']['data_count']} 条\")
print(f\"  GPU 数量: {report['training_config']['num_gpus']} 张\")
print(f\"  训练耗时: {report['training_time']['duration_human']}\")

if 'gpu_stats_summary' in report:
    for gpu_key, gpu_info in report['gpu_stats_summary'].get('gpus', {}).items():
        print(f\"  [{gpu_key}] 显存峰值: {gpu_info['memory_peak_mb']:.0f}/{gpu_info['memory_total_mb']:.0f} MB ({gpu_info['memory_peak_pct']}%)\")
        print(f\"  [{gpu_key}] GPU利用率均值: {gpu_info['gpu_utilization_avg_pct']}%\")

if 'loss_curve' in report:
    lc = report['loss_curve']
    if lc.get('train_loss_start') and lc.get('train_loss_end'):
        print(f\"  Train Loss: {lc['train_loss_start']:.4f} → {lc['train_loss_end']:.4f}\")
    if lc.get('eval_loss_start') and lc.get('eval_loss_end'):
        print(f\"  Eval Loss:  {lc['eval_loss_start']:.4f} → {lc['eval_loss_end']:.4f} (best: {lc['eval_loss_best']:.4f})\")

print(f\"  报告保存至: {report_path}\")
print('=' * 60)
"

echo ""
echo "=========================================="
echo "  训练完成!"
echo "  LoRA 权重保存至: ${OUTPUT_DIR}"
echo "  训练报告: ${TRAIN_REPORT_FILE}"
echo "  GPU 监控: ${GPU_STATS_FILE}"
echo ""
echo "  下一步:"
echo "    1. 评测模型: python train/evaluate_model.py"
echo "    2. 合并权重: python train/merge_lora.py"
echo "=========================================="
