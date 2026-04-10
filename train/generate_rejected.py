"""
为 DPO 偏好数据生成 rejected 回答
使用合并后的 SFT 模型，不加 system prompt，直接让模型回答问题
生成的回答作为低质量的 rejected，与人工标注的 chosen 形成偏好对

用法:
  python train/generate_rejected.py \
      --model outputs/ustc-qa-merged \
      --data data/preference_data.json \
      --output data/preference_data.json \
      --batch-size 8

说明:
  - 使用贪心解码 (do_sample=False)，结果可复现
  - 不加 system prompt，让模型生成"裸回答"作为 rejected
  - 支持断点续传：已有 rejected 的数据会跳过
  - 每处理 50 条自动保存一次，防止中断丢失
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_path):
    """加载合并后的 SFT 模型"""
    print(f"  加载模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )
    model.eval()
    print(f"  模型加载完成")
    return model, tokenizer


def generate_rejected_answer(model, tokenizer, question, max_new_tokens=256):
    """生成低质量的 rejected 回答（不加 system prompt）"""
    # 故意不加 system prompt，让模型生成较通用、不够专业的回答
    messages = [
        {"role": "user", "content": question},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.1,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return answer


def main():
    parser = argparse.ArgumentParser(description="为 DPO 数据生成 rejected 回答")
    parser.add_argument(
        "--model",
        type=str,
        default="outputs/ustc-qa-merged",
        help="合并后的 SFT 模型路径",
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
        "--save-interval",
        type=int,
        default=50,
        help="每处理多少条自动保存一次",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  DPO Rejected 回答生成")
    print("=" * 60)

    # 加载数据
    print("\n[1/3] 加载偏好数据...")
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

    # 加载模型
    print("\n[2/3] 加载模型...")
    model, tokenizer = load_model(args.model)

    # 生成 rejected
    print(f"\n[3/3] 开始生成 rejected 回答...")
    start_time = time.time()
    generated_count = 0
    error_count = 0

    for idx_in_list, data_idx in enumerate(need_generate):
        item = data[data_idx]
        question = item["instruction"]

        try:
            rejected = generate_rejected_answer(model, tokenizer, question)

            # 如果生成的回答太短，用一个简短的通用回答替代
            if len(rejected) < 10:
                rejected = "这个问题我不太清楚，建议你去学校官网查一下。"

            item["rejected"] = rejected
            generated_count += 1

        except Exception as e:
            print(f"  [ERROR] 第 {data_idx} 条生成失败: {e}")
            item["rejected"] = "抱歉，我无法回答这个问题。"
            error_count += 1

        # 进度显示
        progress = idx_in_list + 1
        if progress % 10 == 0 or progress == len(need_generate):
            elapsed = time.time() - start_time
            speed = progress / elapsed if elapsed > 0 else 0
            eta = (len(need_generate) - progress) / speed if speed > 0 else 0
            print(
                f"  进度: {progress}/{len(need_generate)} "
                f"({progress/len(need_generate)*100:.1f}%) "
                f"速度: {speed:.1f} 条/秒 "
                f"预计剩余: {eta/60:.1f} 分钟"
            )

        # 定期保存
        if progress % args.save_interval == 0:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"  [自动保存] 已保存 {progress} 条")

    # 最终保存
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - start_time

    print(f"\n{'=' * 60}")
    print(f"  生成完成!")
    print(f"  成功生成: {generated_count} 条")
    print(f"  生成失败: {error_count} 条")
    print(f"  总耗时: {elapsed/60:.1f} 分钟")
    print(f"  输出文件: {args.output}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
