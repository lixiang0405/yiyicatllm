"""
GPU 监控脚本
后台运行，定时采集显存占用、GPU利用率等指标，保存为 JSON 供训练报告使用。

用法:
  启动: python train/gpu_monitor.py --output outputs/gpu_stats.json &
  停止: kill $(cat /tmp/gpu_monitor.pid)
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def get_gpu_stats():
    """通过 nvidia-smi 采集所有 GPU 的实时指标"""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return []

        stats = []
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 7:
                stats.append({
                    "gpu_index": int(parts[0]),
                    "gpu_name": parts[1],
                    "memory_used_mb": float(parts[2]),
                    "memory_total_mb": float(parts[3]),
                    "memory_used_pct": round(float(parts[2]) / float(parts[3]) * 100, 1),
                    "gpu_utilization_pct": float(parts[4]),
                    "temperature_c": float(parts[5]),
                    "power_w": float(parts[6]),
                })
        return stats
    except Exception:
        return []


def run_monitor(output_path, interval_seconds=5):
    """主监控循环"""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # 写入 PID 文件，方便外部停止
    pid_file = Path("/tmp/gpu_monitor.pid")
    pid_file.write_text(str(os.getpid()))

    records = []
    peak_memory = {}
    running = True

    def handle_stop(signum, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, handle_stop)
    signal.signal(signal.SIGINT, handle_stop)

    start_time = time.time()
    print(f"[GPU Monitor] 启动监控，采样间隔 {interval_seconds}s，PID={os.getpid()}")
    print(f"[GPU Monitor] 输出文件: {output_file}")

    while running:
        elapsed = round(time.time() - start_time, 1)
        gpu_stats = get_gpu_stats()

        if gpu_stats:
            record = {"elapsed_seconds": elapsed, "gpus": gpu_stats}
            records.append(record)

            # 更新峰值显存
            for gpu in gpu_stats:
                gpu_id = gpu["gpu_index"]
                current_mem = gpu["memory_used_mb"]
                if gpu_id not in peak_memory or current_mem > peak_memory[gpu_id]:
                    peak_memory[gpu_id] = current_mem

        time.sleep(interval_seconds)

    # 计算汇总统计
    summary = _compute_summary(records, peak_memory, time.time() - start_time)

    # 保存结果
    result = {"summary": summary, "records": records}
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n[GPU Monitor] 监控结束，共采集 {len(records)} 条记录")
    print(f"[GPU Monitor] 结果保存至: {output_file}")
    _print_summary(summary)

    # 清理 PID 文件
    if pid_file.exists():
        pid_file.unlink()


def _compute_summary(records, peak_memory, total_seconds):
    """计算汇总统计"""
    if not records:
        return {}

    num_gpus = len(records[0]["gpus"]) if records else 0
    summary = {
        "monitoring_duration_seconds": round(total_seconds, 1),
        "num_samples": len(records),
        "num_gpus": num_gpus,
        "gpus": {},
    }

    for gpu_id in range(num_gpus):
        gpu_records = [r["gpus"][gpu_id] for r in records if gpu_id < len(r["gpus"])]
        if not gpu_records:
            continue

        memory_used_list = [g["memory_used_mb"] for g in gpu_records]
        utilization_list = [g["gpu_utilization_pct"] for g in gpu_records]
        power_list = [g["power_w"] for g in gpu_records]

        summary["gpus"][f"gpu_{gpu_id}"] = {
            "name": gpu_records[0]["gpu_name"],
            "memory_total_mb": gpu_records[0]["memory_total_mb"],
            "memory_peak_mb": peak_memory.get(gpu_id, 0),
            "memory_peak_pct": round(peak_memory.get(gpu_id, 0) / gpu_records[0]["memory_total_mb"] * 100, 1),
            "memory_avg_mb": round(sum(memory_used_list) / len(memory_used_list), 1),
            "gpu_utilization_avg_pct": round(sum(utilization_list) / len(utilization_list), 1),
            "gpu_utilization_max_pct": max(utilization_list),
            "power_avg_w": round(sum(power_list) / len(power_list), 1),
            "power_max_w": max(power_list),
        }

    return summary


def _print_summary(summary):
    """打印汇总信息"""
    if not summary:
        return

    print(f"\n{'='*55}")
    print(f"  GPU 监控汇总")
    print(f"{'='*55}")
    print(f"  监控时长: {summary['monitoring_duration_seconds']}s")
    print(f"  采样次数: {summary['num_samples']}")

    for gpu_key, gpu_info in summary.get("gpus", {}).items():
        print(f"\n  [{gpu_key}] {gpu_info['name']}")
        print(f"    显存峰值: {gpu_info['memory_peak_mb']:.0f} / {gpu_info['memory_total_mb']:.0f} MB ({gpu_info['memory_peak_pct']}%)")
        print(f"    显存均值: {gpu_info['memory_avg_mb']:.0f} MB")
        print(f"    GPU利用率均值: {gpu_info['gpu_utilization_avg_pct']}%")
        print(f"    GPU利用率峰值: {gpu_info['gpu_utilization_max_pct']}%")
        print(f"    功耗均值: {gpu_info['power_avg_w']}W")
    print(f"{'='*55}")


def main():
    parser = argparse.ArgumentParser(description="GPU 监控脚本")
    parser.add_argument("--output", type=str, default="outputs/gpu_stats.json", help="输出文件路径")
    parser.add_argument("--interval", type=int, default=5, help="采样间隔(秒)")
    args = parser.parse_args()

    run_monitor(args.output, args.interval)


if __name__ == "__main__":
    main()
