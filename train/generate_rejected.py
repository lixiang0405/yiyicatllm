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


SYSTEM_PROMPT = (
    "你是一个通用的AI助手，请简洁地回答用户的问题。"
    "回答要简短，不需要太详细，给出关键信息即可。"
)


import re


def clean_generated_text(text):
    """清洗生成文本：去除非中英文数字标点的乱码，截断跑偏内容"""
    text = text.strip()

    # 逐字符扫描，遇到连续的非中英文数字标点字符（如泰文、乱码）就截断
    # 允许的字符：中文、英文、数字、常见标点、换行
    allowed_pattern = re.compile(
        r'[\u4e00-\u9fff'        # 中文
        r'\u3000-\u303f'         # 中文标点
        r'\uff00-\uffef'         # 全角字符
        r'a-zA-Z0-9'            # 英文数字
        r'\s'                    # 空白字符
        r'，。！？、；：""''（）【】《》…—·'  # 中文标点
        r',.!?;:\'\"()\[\]{}<>@#$%^&*+=\-_/\\|~`'  # 英文标点
        r']'
    )

    clean_chars = []
    foreign_streak = 0
    for char in text:
        if allowed_pattern.match(char):
            foreign_streak = 0
            clean_chars.append(char)
        else:
            foreign_streak += 1
            # 连续出现 3 个以上非法字符，认为是乱码，截断
            if foreign_streak >= 3:
                # 回退已加入的零散非法字符
                while clean_chars and not allowed_pattern.match(clean_chars[-1]):
                    clean_chars.pop()
                break

    result = ''.join(clean_chars).strip()

    # 去掉末尾不完整的句子（如果最后一个字符不是句号/问号/感叹号/引号）
    end_marks = '。！？"\'）】》…'
    if result and result[-1] not in end_marks:
        # 找最后一个句号位置，截断到那里
        last_mark = -1
        for mark in end_marks:
            pos = result.rfind(mark)
            if pos > last_mark:
                last_mark = pos
        if last_mark > len(result) * 0.5:  # 只有截断不超过一半时才截
            result = result[:last_mark + 1]

    return result


def build_prompts(tokenizer, questions):
    """构建带通用 system prompt 的 chat 格式 prompt（生成质量接近但不够详细的回答）"""
    prompts = []
    for question in questions:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
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
        "--force-regenerate",
        action="store_true",
        help="强制清空所有已有的 rejected，从头重新生成",
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

    # 强制清空已有 rejected
    if args.force_regenerate:
        for item in data:
            item["rejected"] = ""
        print(f"  已清空所有 rejected（--force-regenerate）")

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
        temperature=0.7,            # 适度采样，生成多样但不至于乱码
        top_p=0.85,
        max_tokens=args.max_new_tokens,
        repetition_penalty=1.3,     # 加强重复惩罚，防止重复乱码
        seed=42,                    # 固定种子保证可复现
        stop=["<|im_end|>", "<|endoftext|>", "<|im_start|>", "\nuser", "\nassistant", "\ninstruction"],
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
            rejected = clean_generated_text(output.outputs[0].text)

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