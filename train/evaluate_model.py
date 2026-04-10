"""
SFT 训练后模型评测脚本
评测维度:
  1. 文本重叠: ROUGE-L（词级别 bigram，适合中文长文本）
  2. 领域知识: 关键词命中率（中科大软件学院垂直领域词库）
  3. 事实准确: 事实命中率（URL/数字/机构等关键实体）
  4. 生成质量: 长度合理性、格式质量分
  5. 对比评测: 微调前 vs 微调后（贪心解码，结果可复现）

用法:
  python train/evaluate_model.py \
      --base-model Qwen/Qwen2.5-7B \
      --lora-adapter outputs/ustc-qa-lora \
      --test-data data/qa_pairs.json \
      --output outputs/eval_report.json
"""

# 离线模式：必须在 import transformers/huggingface_hub 之前设置
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse
import json
import re
import time
from pathlib import Path
from typing import List, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def load_model(base_model_path, lora_adapter_path=None):
    """加载模型（支持基座模型和 LoRA 模型，离线模式）"""
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path, trust_remote_code=True, local_files_only=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )

    if lora_adapter_path:
        model = PeftModel.from_pretrained(model, lora_adapter_path)
        model = model.merge_and_unload()

    model.eval()
    return model, tokenizer


def generate_answer(model, tokenizer, question, max_new_tokens=512):
    """生成单个回答"""
    messages = [
        {"role": "system", "content": "你是中科大智能问答助手，请详细、准确地回答用户的问题。回答控制在100-300字以内。"},
        {"role": "user", "content": question},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    start_time = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.2,
        )
    generation_time = time.perf_counter() - start_time

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True)
    output_tokens = len(generated_ids)

    return answer, generation_time, output_tokens


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
    model,
    tokenizer,
    test_data: List[Dict],
    num_samples: int = 50,
    model_label: str = "lora",
):
    """对模型进行全面评测"""
    # 采样测试数据
    import random
    random.seed(42)
    samples = random.sample(test_data, min(num_samples, len(test_data)))

    results = []
    total_tokens = 0
    total_generation_time = 0.0

    print(f"\n  评测 [{model_label}] 模型，共 {len(samples)} 条测试数据...")

    for i, item in enumerate(samples):
        question = item["instruction"]
        reference = item["output"]

        generated, gen_time, output_tokens = generate_answer(model, tokenizer, question)
        total_tokens += output_tokens
        total_generation_time += gen_time

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
            "generation_time_seconds": round(gen_time, 3),
            "output_tokens": output_tokens,
        }
        results.append(result)

        if (i + 1) % 10 == 0:
            print(f"    进度: {i+1}/{len(samples)}")

    # 汇总指标
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
        "avg_generation_time_seconds": round(total_generation_time / num_results, 3),
        "avg_tokens_per_second": round(total_tokens / total_generation_time, 1) if total_generation_time > 0 else 0,
        "total_output_tokens": total_tokens,
    }

    return summary, results


def run_comparison(base_model_path, lora_adapter_path, test_data, num_samples):
    """对比评测：基座模型 vs LoRA 微调模型"""
    all_summaries = {}
    all_details = {}

    # 评测 LoRA 微调模型
    print("\n[1/2] 加载 LoRA 微调模型...")
    lora_model, tokenizer = load_model(base_model_path, lora_adapter_path)
    lora_summary, lora_details = evaluate_model(lora_model, tokenizer, test_data, num_samples, "lora_finetuned")
    all_summaries["lora_finetuned"] = lora_summary
    all_details["lora_finetuned"] = lora_details

    # 释放显存
    del lora_model
    torch.cuda.empty_cache()

    # 评测基座模型
    print("\n[2/2] 加载基座模型 (对比基准)...")
    base_model, tokenizer = load_model(base_model_path, lora_adapter_path=None)
    base_summary, base_details = evaluate_model(base_model, tokenizer, test_data, num_samples, "base_model")
    all_summaries["base_model"] = base_summary
    all_details["base_model"] = base_details

    del base_model
    torch.cuda.empty_cache()

    return all_summaries, all_details


def print_comparison(summaries):
    """打印对比结果"""
    print(f"\n{'='*65}")
    print(f"  模型评测对比报告")
    print(f"{'='*65}")

    header = f"  {'指标':<25} {'基座模型':>15} {'LoRA微调':>15}"
    print(header)
    print(f"  {'-'*55}")

    base = summaries.get("base_model", {})
    lora = summaries.get("lora_finetuned", {})

    metrics = [
        ("ROUGE-L", "avg_rouge_l", True),
        ("关键词命中率", "avg_keyword_hit_rate", True),
        ("事实命中率", "avg_fact_hit_rate", True),
        ("长度比 (生成/参考)", "avg_length_ratio", False),
        ("长度异常比例", "length_abnormal_ratio", False),
        ("格式质量分", "avg_format_score", True),
        ("平均生成速度 (tok/s)", "avg_tokens_per_second", False),
        ("平均生成耗时 (s)", "avg_generation_time_seconds", False),
    ]

    for label, key, higher_is_better in metrics:
        base_val = base.get(key, 0)
        lora_val = lora.get(key, 0)

        if higher_is_better and lora_val > base_val:
            indicator = " ↑"
        elif higher_is_better and lora_val < base_val:
            indicator = " ↓"
        else:
            indicator = ""

        print(f"  {label:<25} {base_val:>15} {lora_val:>13}{indicator}")

    print(f"{'='*65}")


def main():
    parser = argparse.ArgumentParser(description="SFT 训练后模型评测")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-7B", help="基座模型路径")
    parser.add_argument("--lora-adapter", type=str, default="outputs/ustc-qa-lora", help="LoRA 适配器路径")
    parser.add_argument("--test-data", type=str, default="data/new_qa.json", help="测试数据路径")
    parser.add_argument("--num-samples", type=int, default=200, help="评测样本数")
    parser.add_argument("--output", type=str, default="outputs/eval_report.json", help="评测报告输出路径")
    parser.add_argument("--skip-base", action="store_true", help="跳过基座模型评测（省时间）")
    args = parser.parse_args()

    # 加载测试数据
    print("加载测试数据...")
    with open(args.test_data, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    print(f"  测试数据: {len(test_data)} 条")

    if args.skip_base:
        # 只评测 LoRA 模型
        print("\n加载 LoRA 微调模型...")
        model, tokenizer = load_model(args.base_model, args.lora_adapter)
        summary, details = evaluate_model(model, tokenizer, test_data, args.num_samples, "lora_finetuned")
        summaries = {"lora_finetuned": summary}
        all_details = {"lora_finetuned": details}
    else:
        # 对比评测
        summaries, all_details = run_comparison(
            args.base_model, args.lora_adapter, test_data, args.num_samples
        )
        print_comparison(summaries)

    # 保存报告
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "config": {
            "base_model": args.base_model,
            "lora_adapter": args.lora_adapter,
            "test_data": args.test_data,
            "num_samples": args.num_samples,
        },
        "summaries": summaries,
        "sample_details": {k: v[:10] for k, v in all_details.items()},
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n评测报告已保存至: {output_path}")


if __name__ == "__main__":
    main()
