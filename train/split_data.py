"""
从偏好数据中切分公共验证集（SFT + DPO 共用）

用法:
  python train/split_data.py
  python train/split_data.py --pref-data data/preference_data.json --eval-size 300

输出:
  data/eval_data.json       公共验证集（200~300 条）
  data/dpo_train_data.json  DPO 训练集（剩余部分）
"""

import argparse
import json
import random


def main():
    parser = argparse.ArgumentParser(description="从偏好数据中切分公共验证集")
    parser.add_argument(
        "--pref-data",
        type=str,
        default="data/preference_data.json",
        help="偏好数据文件路径",
    )
    parser.add_argument(
        "--eval-output",
        type=str,
        default="data/eval_data.json",
        help="验证集输出路径",
    )
    parser.add_argument(
        "--train-output",
        type=str,
        default="data/dpo_train_data.json",
        help="DPO 训练集输出路径",
    )
    parser.add_argument(
        "--eval-size",
        type=int,
        default=None,
        help="验证集大小（默认 200~300 条，或总量的 10%%）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  切分偏好数据 → 公共验证集 + DPO 训练集")
    print("=" * 60)

    with open(args.pref_data, "r", encoding="utf-8") as f:
        pref_data = json.load(f)

    total = len(pref_data)
    if args.eval_size is not None:
        eval_size = args.eval_size
    else:
        eval_size = min(300, max(200, total // 10))

    random.seed(args.seed)
    random.shuffle(pref_data)

    eval_data = pref_data[:eval_size]
    train_data = pref_data[eval_size:]

    with open(args.eval_output, "w", encoding="utf-8") as f:
        json.dump(eval_data, f, indent=2, ensure_ascii=False)

    with open(args.train_output, "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)

    print(f"  偏好数据: {total} 条")
    print(f"  公共验证集: {eval_size} 条 → {args.eval_output}")
    print(f"  DPO 训练集: {len(train_data)} 条 → {args.train_output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
