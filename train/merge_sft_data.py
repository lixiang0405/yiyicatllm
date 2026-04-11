"""
合并 SFT 训练数据：new_qa.json + preference_data.json 的 chosen 部分
输出统一格式的 SFT 训练集

用法:
  python train/merge_sft_data.py
"""

import json
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="合并 SFT 训练数据")
    parser.add_argument("--new-qa", type=str, default="data/new_qa.json",
                        help="new_qa 数据路径")
    parser.add_argument("--pref-data", type=str, default="data/preference_data.json",
                        help="偏好数据路径")
    parser.add_argument("--output", type=str, default="data/sft_train_data.json",
                        help="合并后的 SFT 训练数据输出路径")
    args = parser.parse_args()

    print("=" * 60)
    print("  合并 SFT 训练数据")
    print("=" * 60)

    # 加载 new_qa 数据
    with open(args.new_qa, "r", encoding="utf-8") as f:
        new_qa_data = json.load(f)
    print(f"  new_qa: {len(new_qa_data)} 条")

    # 加载偏好数据，提取 chosen 作为 output
    with open(args.pref_data, "r", encoding="utf-8") as f:
        pref_data = json.load(f)
    print(f"  preference_data: {len(pref_data)} 条")

    pref_as_sft = []
    for item in pref_data:
        pref_as_sft.append({
            "instruction": item["instruction"],
            "input": item.get("input", ""),
            "output": item["chosen"],
        })

    # 合并并去重（按 instruction 去重，保留第一个）
    seen_instructions = set()
    merged = []

    for item in new_qa_data:
        instruction = item["instruction"].strip()
        if instruction not in seen_instructions:
            seen_instructions.add(instruction)
            merged.append(item)

    duplicates = 0
    for item in pref_as_sft:
        instruction = item["instruction"].strip()
        if instruction not in seen_instructions:
            seen_instructions.add(instruction)
            merged.append(item)
        else:
            duplicates += 1

    # 保存
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"\n  合并结果:")
    print(f"    new_qa 贡献: {len(new_qa_data)} 条")
    print(f"    pref_data 贡献: {len(pref_as_sft) - duplicates} 条 (去重 {duplicates} 条)")
    print(f"    合并总计: {len(merged)} 条")
    print(f"    输出: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
