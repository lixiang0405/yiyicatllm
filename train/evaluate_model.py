"""
SFT/DPO/GRPO 训练后模型评测脚本 (vLLM 加速版)
评测维度:
  1. 文本重叠: ROUGE-L（词级别 bigram，适合中文长文本）
  2. 领域知识: 关键词命中率（中科大软件学院垂直领域词库）
  3. 事实准确: 事实命中率（URL/数字/机构等关键实体）
  4. 生成质量: 长度合理性、格式质量分
  5. 对比评测: 微调前 vs 微调后（贪心解码，结果可复现）

用法:
  # 评测合并后的模型（推荐）
  python train/evaluate_model.py \
      --merged-model /root/autodl-tmp/ustc-qa-dpo-merged \
      --test-data data/eval_data.json

  # 评测 LoRA 适配器（会先合并再用 vLLM 推理）
  python train/evaluate_model.py \
      --base-model Qwen/Qwen2.5-7B \
      --lora-adapter outputs/ustc-qa-lora \
      --test-data data/eval_data.json
"""

# 离线模式：必须在 import transformers/huggingface_hub 之前设置
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse
import gc
import json
import re
import time
from pathlib import Path
from typing import List, Dict, Tuple

import torch

SYSTEM_PROMPT = "你是中科大智能问答助手，请详细、准确地回答用户的问题。回答控制在100-300字以内。"


def load_vllm_model(model_path, tensor_parallel_size=None):
    """加载 vLLM 模型用于批量推理"""
    from vllm import LLM
    from transformers import AutoTokenizer

    if tensor_parallel_size is None:
        tensor_parallel_size = torch.cuda.device_count()

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True
    )

    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        tensor_parallel_size=tensor_parallel_size,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        enforce_eager=True,
    )

    return llm, tokenizer


def merge_lora_to_temp(base_model_path, lora_adapter_path):
    """合并 LoRA 到临时目录，返回合并后的路径"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    import tempfile

    print("  合并 LoRA 权重到临时目录...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path, trust_remote_code=True, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )
    model = PeftModel.from_pretrained(model, lora_adapter_path)
    model = model.merge_and_unload()

    temp_dir = tempfile.mkdtemp(prefix="eval_merged_")
    model.save_pretrained(temp_dir)
    tokenizer.save_pretrained(temp_dir)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    print(f"  合并完成: {temp_dir}")
    return temp_dir


def generate_answers_batch(llm, tokenizer, questions, max_new_tokens=512):
    """用 vLLM 批量生成回答"""
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=max_new_tokens,
        repetition_penalty=1.2,
        stop=["<|im_end|>", "<|endoftext|>", "<|im_start|>"],
    )

    formatted_prompts = []
    for question in questions:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        formatted_prompts.append(formatted)

    start_time = time.perf_counter()
    outputs = llm.generate(formatted_prompts, sampling_params)
    total_time = time.perf_counter() - start_time

    answers = []
    total_tokens = 0
    for output in outputs:
        text = output.outputs[0].text.strip()
        num_tokens = len(output.outputs[0].token_ids)
        answers.append(text)
        total_tokens += num_tokens

    avg_time = total_time / len(questions) if questions else 0
    return answers, total_time, total_tokens, avg_time

def _tokenize_chinese(text):
    """简易中文分词：按字符 bigram 切分，兼顾英文单词完整性"""
    tokens = []
    english_buffer = []
    chars = list(text)
    for char in chars:
        if char.isascii() and char.isalnum():
            english_buffer.append(char)
        else:
            if english_buffer:
                tokens.append("".join(english_buffer))
                english_buffer = []
            if char.strip():
                tokens.append(char)
    if english_buffer:
        tokens.append("".join(english_buffer))
    # 生成 bigram 以捕捉中文词语
    bigrams = []
    for i in range(len(tokens) - 1):
        bigrams.append(tokens[i] + tokens[i + 1])
    return set(tokens) | set(bigrams)

def compute_rouge_l(reference, hypothesis):
    """计算 ROUGE-L F1 分数（基于词级别 bigram 重叠，适合中文长文本）"""
    if not reference.strip() or not hypothesis.strip():
        return 0.0

    ref_tokens = _tokenize_chinese(reference)
    hyp_tokens = _tokenize_chinese(hypothesis)

    if not ref_tokens or not hyp_tokens:
        return 0.0

    overlap = ref_tokens & hyp_tokens
    precision = len(overlap) / len(hyp_tokens)
    recall = len(overlap) / len(ref_tokens)

    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)



def _extract_entities_from_text(text):
    """从文本中动态提取关键实体（专有名词、技术术语、机构名等）"""
    entities = set()

    # 1. 英文专有名词/技术术语（2字符以上的连续英文+数字）
    english_terms = re.findall(r'[A-Za-z][A-Za-z0-9_.+-]{1,30}', text)
    for term in english_terms:
        # 过滤常见虚词
        if term.lower() not in {"the", "and", "for", "with", "from", "that", "this", "are", "was", "not", "can", "will", "has", "have", "but", "also"}:
            entities.add(term)

    # 2. 中文专有名词：书名号/引号内的内容
    quoted = re.findall(r'[《「『"](.*?)[》」』"]', text)
    entities.update(q for q in quoted if 2 <= len(q) <= 20)

    # 3. 中文机构/地点名：xx大学、xx学院、xx医院、xx公司、xx实验室等
    org_patterns = re.findall(r'[\u4e00-\u9fff]{2,10}(?:大学|学院|医院|公司|实验室|研究院|研究所|中心|平台|基金会)', text)
    entities.update(org_patterns)

    # 4. 人名模式：xx教授、xx老师、xx同学等称谓前的名字
    name_patterns = re.findall(r'([\u4e00-\u9fff]{2,4})(?:教授|老师|同学|学长|学姐|院士|博士|导师)', text)
    entities.update(name_patterns)

    # 5. 数字+单位的事实（如"12年"、"32GB"、"985"等）
    num_facts = re.findall(r'\d+(?:年|月|天|人|分|元|%|GB|MB|TB|门|学分|个|条|篇|次|届|期)', text)
    entities.update(num_facts)

    # 6. URL
    urls = re.findall(r'https?://\S+', text)
    entities.update(urls)

    return entities

# 核心高频关键词（精简版，只保留最重要的跨主题通用词）
_CORE_KEYWORDS = {
    "中科大", "中国科学技术大学", "USTC", "软件学院", "合肥", "苏州",
    "医保", "实习", "内推", "校招", "选课", "课程仓库", "开学考",
    "毕业设计", "毕业论文", "答辩", "导师", "培养方案",
}

def compute_keyword_hit_rate(question, reference, generated):
    """计算关键词命中率：核心词表 + 从参考答案动态提取的关键实体"""
    # 混合策略：核心词表中出现在参考答案里的 + 动态提取的实体
    core_hits = {kw for kw in _CORE_KEYWORDS if kw in reference}
    dynamic_entities = _extract_entities_from_text(reference)
    all_keywords = core_hits | dynamic_entities

    if not all_keywords:
        return 1.0

    hit_count = sum(1 for kw in all_keywords if kw in generated)
    return round(hit_count / len(all_keywords), 4)


def compute_length_ratio(reference, generated):
    """计算生成长度与参考长度的比值，并判断是否在合理范围内"""
    ref_len = len(reference)
    gen_len = len(generated)
    if ref_len == 0:
        return 0.0, False
    ratio = round(gen_len / ref_len, 2)
    is_reasonable = 0.3 <= ratio <= 3.0
    return ratio, is_reasonable

def compute_fact_hit_rate(reference, generated):
    """计算事实命中率：从参考答案中提取关键实体（数字、专有名词、URL等），检查生成答案是否包含"""
    # 提取参考答案中的关键事实片段
    fact_patterns = [
        re.findall(r'https?://\S+', reference),                          # URL
        re.findall(r'[\w.]+@[\w.]+', reference),                         # 邮箱
        re.findall(r'\d{4}年|\d+%|\d+学分|\d+门|\d+人', reference),       # 数量事实
        re.findall(r'(?:任职于|就职于|来自|位于|地址[是为]?)\s*(\S{2,15})', reference),  # 机构/地点
    ]

    facts = []
    for pattern_results in fact_patterns:
        facts.extend(pattern_results)

    # 提取引号内的专有名词
    quoted = re.findall(r'[「『"](.*?)[」』"]', reference)
    facts.extend([q for q in quoted if len(q) >= 2])

    # 去重
    facts = list(set(facts))
    if not facts:
        return 1.0, []

    hit_facts = [f for f in facts if f in generated]
    miss_facts = [f for f in facts if f not in generated]
    hit_rate = round(len(hit_facts) / len(facts), 4)
    return hit_rate, miss_facts


def compute_format_score(generated):
    """评估生成回答的格式质量（0-1）"""
    score = 0.0
    max_score = 3.0

    # 有编号列表
    numbered = re.findall(r'(?:^|\n)\s*(?:\d+[.、]|[一二三四五六七八九十]+[、.])', generated)
    if len(numbered) >= 2:
        score += 1.0

    # 有多段落
    paragraphs = [p.strip() for p in generated.split("\n") if p.strip()]
    if len(paragraphs) >= 3:
        score += 1.0

    # 以完整标点结尾
    if generated.strip() and generated.strip()[-1] in "。！？.!?)）":
        score += 1.0

    return round(score / max_score, 2)


def evaluate_model(
    llm,
    tokenizer,
    test_data: List[Dict],
    num_samples: int = 50,
    model_label: str = "lora",
):
    """对模型进行全面评测（vLLM 批量推理）"""
    import random
    random.seed(42)
    samples = random.sample(test_data, min(num_samples, len(test_data)))

    questions = [item["instruction"] for item in samples]
    # 兼容不同数据格式：qa_pairs 用 "output"，preference_data/eval_data 用 "chosen"
    references = [item.get("output") or item.get("chosen", "") for item in samples]

    print(f"\n  评测 [{model_label}] 模型，共 {len(samples)} 条测试数据...")
    print(f"  使用 vLLM 批量推理...")

    answers, total_time, total_tokens, avg_time = generate_answers_batch(
        llm, tokenizer, questions
    )

    results = []
    for question, reference, generated in zip(questions, references, answers):
        rouge_l = compute_rouge_l(reference, generated)
        keyword_hit = compute_keyword_hit_rate(question, reference, generated)
        length_ratio, length_reasonable = compute_length_ratio(reference, generated)
        format_score = compute_format_score(generated)
        fact_hit, miss_facts = compute_fact_hit_rate(reference, generated)

        result = {
            "question": question,
            "reference": reference[:500],
            "generated": generated[:500],
            "rouge_l": rouge_l,
            "keyword_hit_rate": keyword_hit,
            "fact_hit_rate": fact_hit,
            "length_ratio": length_ratio,
            "length_reasonable": length_reasonable,
            "format_score": format_score,
            "miss_facts": miss_facts[:5],
        }
        results.append(result)

    num_results = len(results)
    length_abnormal_count = sum(1 for r in results if not r["length_reasonable"])

    summary = {
        "model_label": model_label,
        "num_samples": num_results,
        "avg_rouge_l": round(sum(r["rouge_l"] for r in results) / num_results, 4),
        "avg_keyword_hit_rate": round(sum(r["keyword_hit_rate"] for r in results) / num_results, 4),
        "avg_fact_hit_rate": round(sum(r["fact_hit_rate"] for r in results) / num_results, 4),
        "avg_length_ratio": round(sum(r["length_ratio"] for r in results) / num_results, 2),
        "length_abnormal_ratio": round(length_abnormal_count / num_results, 4),
        "avg_format_score": round(sum(r["format_score"] for r in results) / num_results, 2),
        "total_generation_time_seconds": round(total_time, 2),
        "avg_tokens_per_second": round(total_tokens / total_time, 1) if total_time > 0 else 0,
        "total_output_tokens": total_tokens,
    }

    return summary, results


def free_vllm(llm):
    """释放 vLLM 模型显存，等待 GPU 进程完全退出"""
    from vllm.distributed.parallel_state import destroy_model_parallel
    try:
        destroy_model_parallel()
    except Exception:
        pass
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    # 等待 vLLM worker 进程完全退出
    time.sleep(5)
    gc.collect()
    torch.cuda.empty_cache()


def _eval_single_model_subprocess(label, model_path, test_data_path, num_samples, result_file):
    """在子进程中评测单个模型，结果写入临时文件"""
    import subprocess
    import sys

    script = f"""
import json, sys
sys.path.insert(0, '.')
from train.evaluate_model import load_vllm_model, evaluate_model

with open("{test_data_path}", "r", encoding="utf-8") as f:
    test_data = json.load(f)

llm, tokenizer = load_vllm_model("{model_path}")
summary, details = evaluate_model(llm, tokenizer, test_data, {num_samples}, "{label}")

result = {{"summary": summary, "details": details[:10]}}
with open("{result_file}", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print("  评测完成，结果已保存")
"""
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(Path(__file__).resolve().parent.parent),
        capture_output=False,
    )
    return proc.returncode

def run_multi_comparison(model_configs, test_data, num_samples, test_data_path=None):
    """多模型对比评测（每个模型在独立子进程中运行，避免 vLLM 显存残留）

    Args:
        model_configs: [(label, path), ...] 模型标签和路径列表
        test_data: 测试数据（仅用于非子进程模式）
        num_samples: 评测样本数
        test_data_path: 测试数据文件路径（子进程模式需要）

    Returns:
        all_summaries: {label: summary}
        all_details: {label: details}
    """
    import tempfile

    all_summaries = {}
    all_details = {}
    total = len(model_configs)

    for idx, (label, model_path) in enumerate(model_configs, 1):
        if not Path(model_path).exists():
            print(f"\n[{idx}/{total}] 跳过 [{label}]（路径不存在: {model_path}）")
            continue

        print(f"\n[{idx}/{total}] 评测 [{label}]: {model_path}")

        # 使用临时文件传递结果
        result_file = tempfile.mktemp(suffix=".json", prefix=f"eval_{label}_")

        returncode = _eval_single_model_subprocess(
            label, model_path, test_data_path, num_samples, result_file
        )

        if returncode != 0:
            print(f"  [ERROR] [{label}] 评测失败 (exit code={returncode})")
            continue

        if Path(result_file).exists():
            with open(result_file, "r", encoding="utf-8") as f:
                result = json.load(f)
            all_summaries[label] = result["summary"]
            all_details[label] = result["details"]
            Path(result_file).unlink()
            print(f"  [{label}] 评测完成 ✓")
        else:
            print(f"  [ERROR] [{label}] 结果文件未生成")

    return all_summaries, all_details

def print_multi_comparison(summaries):
    """打印多模型对比结果"""
    if not summaries:
        print("  没有可用的评测结果")
        return

    labels = list(summaries.keys())
    col_width = 12

    print(f"\n{'='*80}")
    print(f"  多模型对比评测报告")
    print(f"{'='*80}")

    # 表头
    header = f"  {'指标':<20}"
    for label in labels:
        header += f" {label:>{col_width}}"
    print(header)
    print(f"  {'-'*(20 + (col_width + 1) * len(labels))}")

    metrics = [
        ("ROUGE-L", "avg_rouge_l", True),
        ("关键词命中率", "avg_keyword_hit_rate", True),
        ("事实命中率", "avg_fact_hit_rate", True),
        ("长度比", "avg_length_ratio", False),
        ("长度异常比例", "length_abnormal_ratio", False),
        ("格式质量分", "avg_format_score", True),
        ("生成速度(tok/s)", "avg_tokens_per_second", False),
        ("耗时(s)", "total_generation_time_seconds", False),
    ]

    # 找到第一个模型作为基准（通常是基座模型）
    base_label = labels[0]

    for metric_name, key, higher_is_better in metrics:
        row = f"  {metric_name:<20}"
        base_val = summaries[base_label].get(key, 0)

        for label in labels:
            val = summaries[label].get(key, 0)
            if label == base_label or not higher_is_better:
                row += f" {val:>{col_width}}"
            else:
                if val > base_val:
                    row += f" {val:>{col_width-1}}↑"
                elif val < base_val:
                    row += f" {val:>{col_width-1}}↓"
                else:
                    row += f" {val:>{col_width}}"
        print(row)

    print(f"{'='*80}")

    # 打印提升幅度（相对于基座模型）
    if len(labels) >= 2:
        print(f"\n  相对于 [{base_label}] 的提升:")
        key_metrics = [
            ("ROUGE-L", "avg_rouge_l"),
            ("关键词命中率", "avg_keyword_hit_rate"),
            ("事实命中率", "avg_fact_hit_rate"),
            ("格式质量分", "avg_format_score"),
        ]
        for label in labels[1:]:
            print(f"  [{label}]:")
            for metric_name, key in key_metrics:
                base_val = summaries[base_label].get(key, 0)
                cur_val = summaries[label].get(key, 0)
                diff = cur_val - base_val
                pct = (diff / base_val * 100) if base_val > 0 else 0
                sign = "+" if diff >= 0 else ""
                print(f"    {metric_name}: {sign}{diff:.4f} ({sign}{pct:.1f}%)")
        print()


def print_single_summary(summary):
    """打印单个模型的评测结果"""
    print(f"\n{'='*55}")
    print(f"  模型评测报告 [{summary.get('model_label', '')}]")
    print(f"{'='*55}")
    print(f"  {'评测样本数':<25} {summary['num_samples']:>15}")
    print(f"  {'ROUGE-L':<25} {summary['avg_rouge_l']:>15}")
    print(f"  {'关键词命中率':<25} {summary['avg_keyword_hit_rate']:>15}")
    print(f"  {'事实命中率':<25} {summary['avg_fact_hit_rate']:>15}")
    print(f"  {'长度比 (生成/参考)':<25} {summary['avg_length_ratio']:>15}")
    print(f"  {'长度异常比例':<25} {summary['length_abnormal_ratio']:>15}")
    print(f"  {'格式质量分':<25} {summary['avg_format_score']:>15}")
    print(f"  {'生成速度 (tok/s)':<25} {summary['avg_tokens_per_second']:>15}")
    print(f"  {'总生成耗时 (s)':<25} {summary['total_generation_time_seconds']:>15}")
    print(f"{'='*55}")


def main():
    parser = argparse.ArgumentParser(
        description="多模型对比评测 (vLLM 加速)",
        epilog="""
用法示例:
  # 多模型对比评测（推荐）
  python train/evaluate_model.py --models \\
      "基座:/root/autodl-tmp/Qwen2.5-7B" \\
      "LoRA:/root/autodl-tmp/ustc-qa-merged" \\
      "DPO:/root/autodl-tmp/ustc-qa-dpo-merged" \\
      "GRPO:/root/autodl-tmp/ustc-qa-grpo-merged/epoch-3"

  # 单模型评测
  python train/evaluate_model.py --merged-model /root/autodl-tmp/ustc-qa-dpo-merged
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--models", nargs="+", type=str, default=None,
                        help='多模型对比，格式: "标签:模型路径"，如 "基座:/path/to/base" "LoRA:/path/to/lora"')
    parser.add_argument("--merged-model", type=str, default=None,
                        help="合并后的模型路径（单模型评测时使用）")
    parser.add_argument("--base-model", type=str, default=None,
                        help="基座模型路径（用于 LoRA 合并或对比评测）")
    parser.add_argument("--lora-adapter", type=str, default=None,
                        help="LoRA 适配器路径（会先合并再用 vLLM 推理）")
    parser.add_argument("--test-data", type=str, default="data/eval_data.json",
                        help="测试数据路径（默认使用验证集）")
    parser.add_argument("--num-samples", type=int, default=200,
                        help="评测样本数（默认 200，设为 0 则使用全部数据）")
    parser.add_argument("--output", type=str, default="outputs/eval_report.json",
                        help="评测报告输出路径")
    parser.add_argument("--skip-base", action="store_true",
                        help="跳过基座模型对比评测")
    args = parser.parse_args()

    # 加载测试数据
    print("加载测试数据...")
    test_data_path = Path(args.test_data)
    if not test_data_path.exists():
        for fallback in ["data/preference_data.json", "data/qa_pairs.json"]:
            if Path(fallback).exists():
                test_data_path = Path(fallback)
                print(f"  [WARNING] {args.test_data} 不存在，回退使用: {fallback}")
                break
        else:
            print(f"[ERROR] 找不到测试数据文件: {args.test_data}")
            return

    with open(test_data_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    num_samples = args.num_samples if args.num_samples > 0 else len(test_data)
    print(f"  测试数据: {len(test_data)} 条，评测: {min(num_samples, len(test_data))} 条")

    # ========== 多模型对比模式 ==========
    if args.models:
        model_configs = []
        for spec in args.models:
            if ":" in spec:
                label, path = spec.split(":", 1)
            else:
                label = Path(spec).name
                path = spec
            model_configs.append((label.strip(), path.strip()))

        print(f"\n多模型对比评测: {len(model_configs)} 个模型")
        for label, path in model_configs:
            exists = "✓" if Path(path).exists() else "✗"
            print(f"  [{exists}] {label}: {path}")

        summaries, all_details = run_multi_comparison(
            model_configs, test_data, num_samples, test_data_path=str(test_data_path)
        )
        print_multi_comparison(summaries)

    # ========== 单模型 / 双模型对比模式 ==========
    else:
        model_path = args.merged_model
        temp_merged_dir = None

        if model_path is None:
            if args.base_model and args.lora_adapter:
                temp_merged_dir = merge_lora_to_temp(args.base_model, args.lora_adapter)
                model_path = temp_merged_dir
            elif args.base_model:
                model_path = args.base_model
            else:
                for default_path in [
                    "/root/autodl-tmp/ustc-qa-dpo-merged",
                    "/root/autodl-tmp/ustc-qa-merged",
                ]:
                    if Path(default_path).exists():
                        model_path = default_path
                        break
                if model_path is None:
                    print("[ERROR] 未指定模型路径，请使用 --merged-model 或 --models")
                    return

        print(f"  评测模型: {model_path}")

        if args.skip_base or not args.base_model:
            llm, tokenizer = load_vllm_model(model_path)
            summary, details = evaluate_model(
                llm, tokenizer, test_data, num_samples, "finetuned"
            )
            summaries = {"finetuned": summary}
            all_details = {"finetuned": details}
            free_vllm(llm)
            print_single_summary(summary)
        else:
            model_configs = [
                ("base", args.base_model),
                ("finetuned", model_path),
            ]
            summaries, all_details = run_multi_comparison(
                model_configs, test_data, num_samples, test_data_path=str(test_data_path)
            )
            print_multi_comparison(summaries)

        if temp_merged_dir:
            import shutil
            shutil.rmtree(temp_merged_dir, ignore_errors=True)
            print(f"  已清理临时目录: {temp_merged_dir}")

    # 保存报告
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "config": {
            "models": args.models or [args.merged_model],
            "test_data": str(test_data_path),
            "num_samples": num_samples,
        },
        "summaries": summaries,
        "sample_details": {k: v[:10] for k, v in all_details.items()},
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n评测报告已保存至: {output_path}")


if __name__ == "__main__":
    main()