"""
模型量化脚本
使用 transformers 原生 GPTQConfig 进行 GPTQ 量化，无需额外安装 auto-gptq / autoawq。
将 FP16/BF16 模型压缩为 INT4，适配 8GB 显存部署。

依赖：transformers, optimum, torch
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig


CALIBRATION_TEXTS = [
    "中国科学技术大学是一所以前沿科学和高新技术为主的综合性全国重点大学。",
    "中科大少年班创办于1978年，是中国高等教育改革的一面旗帜。",
    "量子信息科学是中科大的优势学科之一，潘建伟院士团队在该领域取得了世界领先的成果。",
    "中科大位于安徽省合肥市，校园环境优美，学术氛围浓厚。",
    "中科大的计算机科学与技术学科在人工智能、量子计算等方向处于国内前列。",
    "中科大毕业生深造率常年保持在70%以上，在全国高校中名列前茅。",
    "中科大拥有合肥微尺度物质科学国家研究中心等多个国家级科研平台。",
    "中科大的物理学、化学、数学等基础学科实力雄厚，多个学科获评A+。",
    "中科大研究生院设有多个一级学科博士点，涵盖理学、工学、管理学等领域。",
    "中科大国家同步辐射实验室是国家大科学装置，为材料科学研究提供重要支撑。",
    "中科大火灾科学国家重点实验室在火灾防治领域具有国际影响力。",
    "中科大与中国科学院深度合作，实行科教融合的人才培养模式。",
    "请介绍一下中国科学技术大学的历史沿革和发展历程。",
    "中科大在量子通信领域取得了哪些重要突破？请详细说明。",
    "中科大的本科教育有什么特色？少年班的选拔机制是怎样的？",
    "请介绍中科大在人工智能和计算机科学方面的研究成果。",
]


def prepare_calibration_dataset(tokenizer, num_samples: int = 128, max_length: int = 512):
    """准备校准数据集，返回 tokenizer 编码后的文本列表"""
    repeated_texts = (CALIBRATION_TEXTS * ((num_samples // len(CALIBRATION_TEXTS)) + 1))[:num_samples]
    return [
        tokenizer(text, return_tensors="pt", padding="max_length",
                  truncation=True, max_length=max_length)["input_ids"].squeeze(0)
        for text in repeated_texts
    ]


def quantize_gptq_native(model_path: str, output_path: str, bits: int = 4):
    """使用 transformers 原生 GPTQConfig 进行量化（无需 auto-gptq）"""
    print(f"[GPTQ] 加载 tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[GPTQ] 准备校准数据...")
    calibration_dataset = prepare_calibration_dataset(tokenizer)

    gptq_config = GPTQConfig(
        bits=bits,
        group_size=128,
        desc_act=True,
        dataset=calibration_dataset,
        tokenizer=tokenizer,
        damp_percent=0.1,
    )

    print(f"[GPTQ] 加载模型并量化 (bits={bits})，这可能需要 10-30 分钟...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=gptq_config,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[GPTQ] 保存量化模型...")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    total_size = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file())
    print(f"[GPTQ] ✅ 量化完成! 模型大小: {total_size / 1024**3:.2f} GB")
    print(f"[GPTQ] 保存至: {output_dir}")


def quantize_bitsandbytes(model_path: str, output_path: str, bits: int = 4):
    """使用 bitsandbytes 进行量化（备选方案，零额外依赖）"""
    from transformers import BitsAndBytesConfig

    print(f"[BNB] 加载模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"[BNB] 加载并量化模型 (NF4)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map="auto",
    )

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    total_size = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file())
    print(f"[BNB] ✅ 量化完成! 模型大小: {total_size / 1024**3:.2f} GB")
    print(f"[BNB] 保存至: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="模型量化 (GPTQ / BNB)")
    parser.add_argument(
        "--model-path",
        type=str,
        default="/root/autodl-tmp/ustc-qa-dpo-merged",
        help="合并后模型的路径",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="/root/autodl-tmp/ustc-qa-dpo-quantized",
        help="量化后模型的输出路径",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="gptq",
        choices=["gptq", "bnb"],
        help="量化方法: gptq (推荐，vLLM 兼容) 或 bnb (备选)",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=4,
        choices=[4, 8],
        help="量化位数",
    )
    args = parser.parse_args()

    output_path = f"{args.output_path}-{args.method}-int{args.bits}"

    print("=" * 50)
    print(f"  模型量化")
    print(f"  方法: {args.method.upper()}")
    print(f"  位数: INT{args.bits}")
    print(f"  输入: {args.model_path}")
    print(f"  输出: {output_path}")
    print("=" * 50)

    if args.method == "gptq":
        quantize_gptq_native(args.model_path, output_path, args.bits)
    else:
        quantize_bitsandbytes(args.model_path, output_path, args.bits)

    print(f"\n下一步:")
    print(f"  1. 验证: python3 -c \"from transformers import AutoModelForCausalLM; m = AutoModelForCausalLM.from_pretrained('{output_path}', device_map='auto'); print('✅ 加载成功')\"")
    print(f"  2. 部署: bash deploy/serve.sh --model {output_path}")


if __name__ == "__main__":
    main()