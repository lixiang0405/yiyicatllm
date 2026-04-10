"""
为 DPO 偏好数据生成 rejected 回答
使用 vLLM 批量推理加速，不加 system prompt，直接让模型回答问题
生成的回答作为低质量的 rejected，与人工标注的 chosen 形成偏好对

用法:
  python train/generate_rejected.py \
      --model /root/autodl-tmp/ustc-qa-merged \
      --data data/preference_data.json \
      --output data/preference_data.json \
      --batch-size 256

说明:
  - 使用 vLLM 批量推理，速度比逐条 HF generate 快 10-20 倍
  - 使用贪心解码 (temperature=0)，结果可复现
  - 不加 system prompt，让模型生成"裸回答"作为 rejected
  - 支持断点续传：已有 rejected 的数据会跳过
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse
import json
import time
from pathlib import Path


def build_prompts(tokenizer, questions):
    """构建不带 system prompt 的 chat 格式 prompt"""
    prompts = []
    for question in questions:
        messages = [{"role": "user", "content": question}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(prompt)
    return prompts


def main():
    parser = argparse.ArgumentParser(description="为 DPO 数据生成 rejected 回答 (vLLM 加速)")
    parser.add_argument(
        "--model",
        type=str,
        default="/root/autodl-tmp/ustc-qa-merged",
        help="合并后的 SFT 模型路径（数据盘）",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/preference_data.json",
        help="偏好数据文件路径",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/preference_data.json",
        help="输出文件路径（默认覆盖原文件）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="vLLM 每批处理的 prompt 数量",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="每条回答的最大生成长度",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=None,
        help="vLLM tensor parallel 大小（默认自动检测 GPU 数量）",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  DPO Rejected 回答生成 (vLLM 加速)")
    print("=" * 60)

    # 加载数据
    print("\n[1/4] 加载偏好数据...")
    with open(args.data, "r", encoding="utf-8") as f:
        data = json.load(f)
    total = len(data)

    # 统计需要生成的数量
    need_generate = [i for i, item in enumerate(data) if not item.get("rejected", "").strip()]
    already_done = total - len(need_generate)
    print(f"  总数据: {total} 条")
    print(f"  已有 rejected: {already_done} 条 (跳过)")
    print(f"  需要生成: {len(need_generate)} 条")

    if not need_generate:
        print("\n  所有数据已有 rejected，无需生成！")
        return

    # 自动检测 GPU 数量
    import torch
    if args.tensor_parallel_size is None:
        args.tensor_parallel_size = torch.cuda.device_count()
    print(f"  Tensor Parallel: {args.tensor_parallel_size} 张 GPU")

    # 加载 vLLM 模型
    print("\n[2/4] 加载 vLLM 模型...")
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True
    )
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
    )
    sampling_params = SamplingParams(
        temperature=0,              # 贪心解码，结果可复现
        max_tokens=args.max_new_tokens,
        repetition_penalty=1.1,
    )
    print(f"  vLLM 模型加载完成")

    # 构建 prompt
    print("\n[3/4] 构建 prompt...")
    questions = [data[i]["instruction"] for i in need_generate]
    prompts = build_prompts(tokenizer, questions)
    print(f"  构建完成: {len(prompts)} 条 prompt")

    # 分批推理
    print(f"\n[4/4] 开始批量生成 rejected 回答...")
    start_time = time.time()
    generated_count = 0
    error_count = 0

    for batch_start in range(0, len(prompts), args.batch_size):
        batch_end = min(batch_start + args.batch_size, len(prompts))
        batch_prompts = prompts[batch_start:batch_end]

        outputs = llm.generate(batch_prompts, sampling_params)

        for offset, output in enumerate(outputs):
            global_idx = batch_start + offset
            data_idx = need_generate[global_idx]
            rejected = output.outputs[0].text.strip()

            if len(rejected) < 10:
                rejected = "这个问题我不太清楚，建议你去学校官网查一下。"

            data[data_idx]["rejected"] = rejected
            generated_count += 1

        # 每批结束后保存 + 打印进度
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        elapsed = time.time() - start_time
        done = batch_end
        speed = done / elapsed if elapsed > 0 else 0
        eta = (len(prompts) - done) / speed if speed > 0 else 0
        print(
            f"  进度: {done}/{len(prompts)} "
            f"({done/len(prompts)*100:.1f}%) "
            f"速度: {speed:.1f} 条/秒 "
            f"预计剩余: {eta/60:.1f} 分钟"
        )

    elapsed = time.time() - start_time

    print(f"\n{'=' * 60}")
    print(f"  生成完成!")
    print(f"  成功生成: {generated_count} 条")
    print(f"  生成失败: {error_count} 条")
    print(f"  总耗时: {elapsed/60:.1f} 分钟")
    print(f"  平均速度: {generated_count/elapsed:.1f} 条/秒")
    print(f"  输出文件: {args.output}")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    main()