#!/usr/bin/env python3
"""
DPO 偏好数据清洗脚本
功能：
1. 合并 dpo_line_data 目录下所有 JSON 文件
2. 检查数据格式完整性（instruction/input/chosen/rejected 四个字段）
3. 检查 chosen 长度（100-300字），过短/过长的进行标记或截断
4. 检查 rejected 是否为空
5. 检查限定词（中科大/软件学院等）是否充足
6. 去重：完全重复 + 高度相似（基于问题的编辑距离）
7. 输出清洗后的 preference_data.json 和质量报告
"""

import json
import os
import re
from collections import Counter
from difflib import SequenceMatcher

# ============ 配置 ============
DPO_LINE_DIR = "/Users/yiyicat/yiyicat-llm/data/dpo_line_data"
OUTPUT_FILE = "/Users/yiyicat/yiyicat-llm/data/preference_data.json"
REPORT_FILE = "/Users/yiyicat/yiyicat-llm/data/dpo_quality_report.txt"

MIN_CHOSEN_LEN = 100
MAX_CHOSEN_LEN = 300
MIN_REJECTED_LEN = 10
SIMILARITY_THRESHOLD = 0.85  # 问题相似度阈值

# 限定词列表
DOMAIN_KEYWORDS = [
    "中科大", "中国科学技术大学", "科大", "USTC",
    "软件学院", "软院", "科软",
    "苏高院", "苏州高等研究院", "苏州校区",
    "纳米学院", "生命医学",
]


def load_all_data():
    """加载 dpo_line_data 目录下所有 JSON 文件"""
    all_data = []
    files = sorted([f for f in os.listdir(DPO_LINE_DIR) if f.endswith(".json")])
    for filename in files:
        filepath = os.path.join(DPO_LINE_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"  加载 {filename}: {len(data)} 条")
        for item in data:
            item["_source"] = filename
        all_data.extend(data)
    return all_data


def check_fields(data):
    """检查字段完整性"""
    required_fields = ["instruction", "input", "chosen", "rejected"]
    invalid = []
    valid = []
    for item in data:
        missing = [f for f in required_fields if f not in item]
        if missing:
            invalid.append((item, f"缺少字段: {missing}"))
        else:
            valid.append(item)
    return valid, invalid


def check_chosen_length(data):
    """检查 chosen 长度，过长的智能截断"""
    good = []
    too_short = []
    truncated = []

    for item in data:
        chosen_len = len(item["chosen"])
        if chosen_len < MIN_CHOSEN_LEN:
            too_short.append(item)
        elif chosen_len > MAX_CHOSEN_LEN:
            # 智能截断：在最后一个句号处截断
            text = item["chosen"][:MAX_CHOSEN_LEN]
            last_period = max(text.rfind("。"), text.rfind("！"), text.rfind("？"))
            if last_period > MIN_CHOSEN_LEN:
                item["chosen"] = text[: last_period + 1]
            else:
                item["chosen"] = text
            truncated.append((item, chosen_len))
            good.append(item)
        else:
            good.append(item)

    return good, too_short, truncated


def check_rejected(data):
    """检查 rejected 是否为空或过短（仅统计，不过滤）"""
    has_rejected = []
    empty_rejected = []

    for item in data:
        rejected = item.get("rejected", "").strip()
        if len(rejected) < MIN_REJECTED_LEN:
            empty_rejected.append(item)
        else:
            has_rejected.append(item)

    # 返回全部数据（不过滤），仅做统计
    return data, empty_rejected


def check_domain_keywords(data):
    """检查限定词覆盖情况"""
    with_keywords = []
    without_keywords = []

    for item in data:
        full_text = item["instruction"] + item["chosen"]
        has_keyword = any(kw in full_text for kw in DOMAIN_KEYWORDS)
        if has_keyword:
            with_keywords.append(item)
        else:
            without_keywords.append(item)

    return with_keywords, without_keywords


def deduplicate_exact(data):
    """完全去重：基于 instruction 完全相同"""
    seen = {}
    unique = []
    duplicates = []

    for item in data:
        key = item["instruction"].strip()
        if key in seen:
            duplicates.append(item)
        else:
            seen[key] = item
            unique.append(item)

    return unique, duplicates


def deduplicate_similar(data):
    """高度相似去重：基于问题的编辑距离"""
    unique = []
    similar_groups = []

    for item in data:
        question = item["instruction"].strip()
        is_similar = False
        for existing in unique:
            existing_q = existing["instruction"].strip()
            ratio = SequenceMatcher(None, question, existing_q).ratio()
            if ratio >= SIMILARITY_THRESHOLD:
                similar_groups.append((item, existing, ratio))
                is_similar = True
                break
        if not is_similar:
            unique.append(item)

    return unique, similar_groups


def generate_report(report_lines):
    """生成质量报告"""
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\n质量报告已保存: {REPORT_FILE}")


def main():
    report = []
    report.append("=" * 70)
    report.append("  DPO 偏好数据质量报告")
    report.append("=" * 70)

    # Step 1: 加载所有数据
    print("\n[1/7] 加载数据...")
    all_data = load_all_data()
    report.append(f"\n[1] 原始数据总量: {len(all_data)} 条")

    # Step 2: 检查字段完整性
    print(f"\n[2/7] 检查字段完整性...")
    valid_data, invalid_data = check_fields(all_data)
    report.append(f"\n[2] 字段检查:")
    report.append(f"  有效: {len(valid_data)} 条")
    report.append(f"  无效: {len(invalid_data)} 条")
    if invalid_data:
        for item, reason in invalid_data[:10]:
            report.append(f"    - {reason}: {item.get('instruction', '?')[:60]}")

    # Step 3: 检查 rejected 是否为空
    print(f"\n[3/7] 检查 rejected 字段...")
    valid_data, empty_rejected = check_rejected(valid_data)
    report.append(f"\n[3] rejected 检查:")
    report.append(f"  有rejected: {len(valid_data) - len(empty_rejected)} 条")
    report.append(f"  rejected为空: {len(empty_rejected)} 条 (保留，后续用SFT模型生成)")
    report.append(f"  注意: rejected 为空的数据需要后续补充低质量回答才能用于DPO训练")

    # Step 4: 检查 chosen 长度
    print(f"\n[4/7] 检查 chosen 长度...")
    valid_data, too_short, truncated = check_chosen_length(valid_data)
    report.append(f"\n[4] chosen 长度检查 (要求 {MIN_CHOSEN_LEN}-{MAX_CHOSEN_LEN} 字):")
    report.append(f"  合格: {len(valid_data)} 条")
    report.append(f"  过短(<{MIN_CHOSEN_LEN}字): {len(too_short)} 条 (已移除)")
    report.append(f"  过长(>{MAX_CHOSEN_LEN}字): {len(truncated)} 条 (已截断)")
    if too_short:
        for item in too_short[:10]:
            report.append(
                f"    - [{len(item['chosen'])}字] Q: {item['instruction'][:60]}..."
            )

    # Step 5: 完全去重
    print(f"\n[5/7] 完全去重...")
    valid_data, exact_dups = deduplicate_exact(valid_data)
    report.append(f"\n[5] 完全去重:")
    report.append(f"  去重后: {len(valid_data)} 条")
    report.append(f"  重复: {len(exact_dups)} 条")
    if exact_dups:
        for item in exact_dups[:10]:
            report.append(f"    - Q: {item['instruction'][:60]}...")

    # Step 6: 高度相似去重
    print(f"\n[6/7] 高度相似去重 (阈值={SIMILARITY_THRESHOLD})...")
    valid_data, similar_groups = deduplicate_similar(valid_data)
    report.append(f"\n[6] 相似去重 (阈值={SIMILARITY_THRESHOLD}):")
    report.append(f"  去重后: {len(valid_data)} 条")
    report.append(f"  相似组: {len(similar_groups)} 组")
    if similar_groups:
        for item, existing, ratio in similar_groups[:20]:
            report.append(f"    - [{ratio:.2f}] Q1: {item['instruction'][:50]}...")
            report.append(f"             Q2: {existing['instruction'][:50]}...")

    # Step 7: 检查限定词覆盖
    print(f"\n[7/7] 检查限定词覆盖...")
    with_kw, without_kw = check_domain_keywords(valid_data)
    report.append(f"\n[7] 限定词覆盖:")
    report.append(
        f"  含限定词: {len(with_kw)} 条 ({len(with_kw)/len(valid_data)*100:.1f}%)"
    )
    report.append(
        f"  无限定词: {len(without_kw)} 条 ({len(without_kw)/len(valid_data)*100:.1f}%)"
    )
    if without_kw:
        report.append(f"  （以下数据不含中科大/软件学院等限定词，但仍保留）")
        for item in without_kw[:15]:
            report.append(f"    - Q: {item['instruction'][:70]}...")

    # 统计最终数据的长度分布
    chosen_lens = [len(item["chosen"]) for item in valid_data]
    rejected_lens = [len(item["rejected"]) for item in valid_data]
    report.append(f"\n{'=' * 70}")
    report.append(f"  最终数据统计")
    report.append(f"{'=' * 70}")
    report.append(f"  总条数: {len(valid_data)}")
    report.append(
        f"  chosen 长度: 平均 {sum(chosen_lens)/len(chosen_lens):.0f} 字, "
        f"最短 {min(chosen_lens)} 字, 最长 {max(chosen_lens)} 字"
    )
    report.append(
        f"  rejected 长度: 平均 {sum(rejected_lens)/len(rejected_lens):.0f} 字, "
        f"最短 {min(rejected_lens)} 字, 最长 {max(rejected_lens)} 字"
    )

    # 长度分布
    len_ranges = [(100, 150), (150, 200), (200, 250), (250, 300)]
    report.append(f"\n  chosen 长度分布:")
    for low, high in len_ranges:
        count = sum(1 for l in chosen_lens if low <= l < high)
        report.append(f"    {low}-{high}字: {count} 条 ({count/len(chosen_lens)*100:.1f}%)")

    # 清理临时字段
    for item in valid_data:
        item.pop("_source", None)

    # 保存最终数据
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(valid_data, f, indent=2, ensure_ascii=False)

    report.append(f"\n  输出文件: {OUTPUT_FILE}")
    report.append(f"{'=' * 70}")

    # 输出报告
    generate_report(report)

    # 打印摘要
    print(f"\n{'=' * 60}")
    print(f"  清洗完成!")
    print(f"  原始数据: {len(all_data)} 条")
    print(f"  rejected为空(保留): {len(empty_rejected)} 条")
    print(f"  过短移除: {len(too_short)} 条")
    print(f"  完全重复移除: {len(exact_dups)} 条")
    print(f"  相似重复移除: {len(similar_groups)} 条")
    print(f"  过长截断: {len(truncated)} 条")
    print(f"  最终数据: {len(valid_data)} 条")
    print(f"  含限定词: {len(with_kw)} 条 ({len(with_kw)/len(valid_data)*100:.1f}%)")
    print(f"  输出: {OUTPUT_FILE}")
    print(f"  报告: {REPORT_FILE}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
