"""
从 LLaMA-Factory DPO 训练输出中提取指标，生成训练报告

用法:
  python train/generate_dpo_report.py \
      --output-dir outputs/ustc-qa-dpo \
      --report-dir report/dpo_report
"""

import os
import sys
import json
import time
import platform
import argparse
from pathlib import Path

def load_trainer_log(output_dir: str) -> list:
    """加载 LLaMA-Factory 的 trainer_log.jsonl"""
    log_path = Path(output_dir) / "trainer_log.jsonl"
    if not log_path.exists():
        print(f"[WARNING] 训练日志不存在: {log_path}")
        return []
    entries = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def load_trainer_state(output_dir: str) -> dict:
    """加载 trainer_state.json（包含 eval 指标）"""
    state_path = Path(output_dir) / "trainer_state.json"
    if not state_path.exists():
        return {}
    with open(state_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_all_results(output_dir: str) -> dict:
    """加载 all_results.json"""
    results_path = Path(output_dir) / "all_results.json"
    if not results_path.exists():
        return {}
    with open(results_path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_report(output_dir: str, report_dir: str, train_config: str = None):
    """生成 DPO 训练报告"""
    output_path = Path(output_dir)
    report_path = Path(report_dir)
    report_path.mkdir(parents=True, exist_ok=True)

    # 加载数据
    trainer_log = load_trainer_log(output_dir)
    trainer_state = load_trainer_state(output_dir)
    all_results = load_all_results(output_dir)

    if not trainer_log:
        print("[ERROR] 没有找到训练日志，无法生成报告")
        return

    # 提取训练时间
    first_entry = trainer_log[0]
    last_entry = trainer_log[-1]
    elapsed_time = last_entry.get("elapsed_time", "未知")

    # 提取 loss 曲线
    train_losses = [e["loss"] for e in trainer_log if "loss" in e]

    # 提取 eval 指标（从 trainer_state 的 log_history 中）
    eval_entries = []
    if "log_history" in trainer_state:
        for entry in trainer_state["log_history"]:
            if "eval_loss" in entry:
                eval_entries.append(entry)

    eval_losses = [e["eval_loss"] for e in eval_entries]

    # 收集 GPU 信息
    gpu_info = []
    try:
        import torch
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpu_info.append({
                "index": i,
                "name": props.name,
                "total_memory_gb": round(props.total_mem / 1024**3, 1),
            })
    except Exception:
        gpu_info = [{"note": "GPU 信息不可用（非 CUDA 环境）"}]

    # 加载训练配置
    config_data = {}
    if train_config and Path(train_config).exists():
        try:
            import yaml
            with open(train_config, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)
        except ImportError:
            with open(train_config, "r", encoding="utf-8") as f:
                config_data = {"raw": f.read()}

    # 构建报告
    report = {
        "environment": {
            "python_version": platform.python_version(),
            "os": platform.platform(),
            "gpus": gpu_info,
        },
        "training_time": {
            "elapsed_time": elapsed_time,
            "total_steps": last_entry.get("total_steps", 0),
        },
        "training_config": {
            "stage": "dpo",
            "framework": "LLaMA-Factory",
            **{k: v for k, v in config_data.items() if k != "raw"},
        },
        "training_results": {
            "total_steps": last_entry.get("total_steps", 0),
            "train_loss_start": round(train_losses[0], 4) if train_losses else None,
            "train_loss_end": round(train_losses[-1], 4) if train_losses else None,
            "train_loss_min": round(min(train_losses), 4) if train_losses else None,
            "eval_loss_start": round(eval_losses[0], 4) if eval_losses else None,
            "eval_loss_end": round(eval_losses[-1], 4) if eval_losses else None,
            "eval_loss_min": round(min(eval_losses), 4) if eval_losses else None,
            "eval_loss_best_step": eval_entries[eval_losses.index(min(eval_losses))].get("step") if eval_losses else None,
            "num_train_entries": len(train_losses),
            "num_eval_entries": len(eval_losses),
        },
        "all_results": all_results,
        "training_history": {
            "train_log": trainer_log,
            "eval_log": eval_entries,
        },
    }

    # 保存报告
    report_file = report_path / "dpo_train_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"DPO 训练报告已保存: {report_file}")

    # 复制关键文件到报告目录
    files_to_copy = [
        "trainer_log.jsonl",
        "trainer_state.json",
        "all_results.json",
    ]
    import shutil
    for filename in files_to_copy:
        src = output_path / filename
        if src.exists():
            dst = report_path / f"dpo_{filename}"
            shutil.copy2(str(src), str(dst))

    # 打印摘要
    print("\n" + "=" * 50)
    print("  DPO 训练报告摘要")
    print("=" * 50)
    print(f"  训练时长: {elapsed_time}")
    print(f"  总步数: {last_entry.get('total_steps', '?')}")
    if train_losses:
        print(f"  Train Loss: {train_losses[0]:.4f} → {train_losses[-1]:.4f}")
    if eval_losses:
        print(f"  Eval Loss:  {eval_losses[0]:.4f} → {eval_losses[-1]:.4f} (最低: {min(eval_losses):.4f})")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成 DPO 训练报告")
    parser.add_argument("--output-dir", type=str, default="outputs/ustc-qa-dpo",
                        help="LLaMA-Factory DPO 训练输出目录")
    parser.add_argument("--report-dir", type=str, default="report/dpo_report",
                        help="报告保存目录")
    parser.add_argument("--train-config", type=str, default="train/train_dpo.yaml",
                        help="DPO 训练配置文件")
    args = parser.parse_args()

    generate_report(args.output_dir, args.report_dir, args.train_config)
