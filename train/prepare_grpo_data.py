"""
准备 GRPO 训练数据
将 SFT + DPO 数据转换为 veRL GRPO 训练所需的 prompt-only parquet 格式

veRL 要求:
  - 数据格式: parquet
  - 字段: prompt (chat 格式的 JSON 字符串)
  - 示例: {"prompt": "[{\"role\": \"user\", \"content\": \"...\"}]"}

数据来源:
  - new_qa.json (SFT 训练数据, ~2940 条)
  - preference_data.json (DPO 偏好数据, ~1262 条)
  合并去重后提取所有不重复的问题作为 GRPO prompt
"""

import json
import argparse
from pathlib import Path


def load_questions_from_sft(filepath):
    """从 SFT 数据中提取问题"""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    questions = []
    for item in data:
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        content = f"{instruction}\n{input_text}" if input_text else instruction
        questions.append(content.strip())
    return questions


def load_questions_from_preference(filepath):
    """从偏好数据中提取问题"""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    questions = []
    for item in data:
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        content = f"{instruction}\n{input_text}" if input_text else instruction
        questions.append(content.strip())
    return questions


def convert_to_grpo_parquet(questions, output_path):
    """将问题列表转换为 veRL GRPO parquet 格式"""
    try:
        import pandas as pd
    except ImportError:
        print("[ERROR] 需要安装 pandas: pip install pandas pyarrow")
        raise

    # 构建 chat 格式的 prompt
    records = []
    for question in questions:
        prompt = json.dumps(
            [{"role": "user", "content": question}], ensure_ascii=False
        )
        records.append({"prompt": prompt})

    df = pd.DataFrame(records)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_file, index=False)

    print(f"  转换完成: {len(df)} 条 prompts")
    print(f"  输出文件: {output_file}")
    return df


def main():
    parser = argparse.ArgumentParser(description="准备 GRPO 训练数据")
    parser.add_argument(
        "--sft-data",
        type=str,
        default="data/new_qa.json",
        help="SFT 格式的训练数据路径",
    )
    parser.add_argument(
        "--pref-data",
        type=str,
        default="data/preference_data.json",
        help="DPO 偏好数据路径",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/grpo_prompts.parquet",
        help="输出的 GRPO 数据路径 (parquet 格式)",
    )
    args = parser.parse_args()

    print("[1/3] 加载数据...")
    all_questions = []

    if Path(args.sft_data).exists():
        sft_questions = load_questions_from_sft(args.sft_data)
        print(f"  SFT 数据: {len(sft_questions)} 条")
        all_questions.extend(sft_questions)

    if Path(args.pref_data).exists():
        pref_questions = load_questions_from_preference(args.pref_data)
        print(f"  偏好数据: {len(pref_questions)} 条")
        all_questions.extend(pref_questions)

    if not all_questions:
        print("[ERROR] 未找到任何数据文件")
        return

    print(f"\n[2/3] 去重...")
    unique_questions = list(dict.fromkeys(all_questions))
    print(f"  去重前: {len(all_questions)} 条")
    print(f"  去重后: {len(unique_questions)} 条")

    print(f"\n[3/3] 转换为 parquet...")
    convert_to_grpo_parquet(unique_questions, args.output)


if __name__ == "__main__":
    main()