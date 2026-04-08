"""
准备 GRPO 训练数据
将 SFT 格式的数据转换为 veRL GRPO 训练所需的 prompt-only 格式

GRPO 训练只需要 prompt，不需要 response:
  - SFT 数据: {"instruction": "...", "input": "...", "output": "..."}
  - GRPO 数据: {"prompt": [{"role": "user", "content": "..."}]}

模型会自己生成多个 response (rollout)，然后用奖励函数打分，
通过组内相对排名计算优势函数来更新策略。
"""

import json
import argparse
from pathlib import Path


def convert_sft_to_grpo(sft_data_path: str, output_path: str):
    """将 SFT 数据转换为 GRPO prompt-only 格式"""

    with open(sft_data_path, 'r', encoding='utf-8') as f:
        sft_data = json.load(f)

    grpo_data = []
    for item in sft_data:
        instruction = item.get('instruction', '')
        input_text = item.get('input', '')

        # 构建 prompt
        if input_text:
            content = f'{instruction}\n{input_text}'
        else:
            content = instruction

        grpo_item = {
            'prompt': [
                {'role': 'user', 'content': content}
            ]
        }
        grpo_data.append(grpo_item)

    # 保存
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(grpo_data, f, ensure_ascii=False, indent=2)

    print(f'转换完成: {len(grpo_data)} 条数据')
    print(f'输出文件: {output_file}')


def main():
    parser = argparse.ArgumentParser(description='准备 GRPO 训练数据')
    parser.add_argument(
        '--sft-data', type=str,
        default='data/sample_data.json',
        help='SFT 格式的训练数据路径'
    )
    parser.add_argument(
        '--output', type=str,
        default='data/grpo_prompts.json',
        help='输出的 GRPO 数据路径'
    )
    args = parser.parse_args()

    convert_sft_to_grpo(args.sft_data, args.output)


if __name__ == '__main__':
    main()
