"""
模型量化脚本
支持 GPTQ 和 AWQ 两种量化方式，将 FP16/BF16 模型压缩为 INT4，适配 8GB 显存部署。
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def quantize_gptq(model_path: str, output_path: str, bits: int = 4):
    """使用 GPTQ 进行量化"""
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

    print(f"[GPTQ] 加载模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # 量化配置
    quantize_config = BaseQuantizeConfig(
        bits=bits,
        group_size=128,
        desc_act=True,
        damp_percent=0.1,
    )

    # 加载模型
    model = AutoGPTQForCausalLM.from_pretrained(
        model_path,
        quantize_config=quantize_config,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )

    # 准备校准数据（使用简单的中文文本）
    calibration_texts = [
        "中国科学技术大学是一所以前沿科学和高新技术为主的综合性全国重点大学。",
        "中科大少年班创办于1978年，是中国高等教育改革的一面旗帜。",
        "量子信息科学是中科大的优势学科之一，潘建伟院士团队在该领域取得了世界领先的成果。",
        "中科大位于安徽省合肥市，校园环境优美，学术氛围浓厚。",
        "中科大的计算机科学与技术学科在人工智能、量子计算等方向处于国内前列。",
        "中科大毕业生深造率常年保持在70%以上，在全国高校中名列前茅。",
        "中科大拥有合肥微尺度物质科学国家研究中心等多个国家级科研平台。",
        "中科大的物理学、化学、数学等基础学科实力雄厚，多个学科获评A+。",
    ]

    calibration_data = [
        tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        for text in calibration_texts
    ]

    # 执行量化
    print(f"[GPTQ] 开始量化 (bits={bits})...")
    model.quantize(calibration_data)

    # 保存
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_quantized(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    total_size = sum(f.stat().st_size for f in output_dir.iterdir() if f.is_file())
    print(f"[GPTQ] ✅ 量化完成! 模型大小: {total_size / 1024**3:.2f} GB")
    print(f"[GPTQ] 保存至: {output_dir}")


def quantize_awq(model_path: str, output_path: str, bits: int = 4):
    """使用 AWQ 进行量化"""
    from awq import AutoAWQForCausalLM

    print(f"[AWQ] 加载模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # 加载模型
    model = AutoAWQForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        safetensors=True,
    )

    # 量化配置
    quant_config = {
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": bits,
        "version": "GEMM",
    }

    # 执行量化
    print(f"[AWQ] 开始量化 (bits={bits})...")
    model.quantize(tokenizer, quant_config=quant_config)

    # 保存
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_quantized(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    total_size = sum(f.stat().st_size for f in output_dir.iterdir() if f.is_file())
    print(f"[AWQ] ✅ 量化完成! 模型大小: {total_size / 1024**3:.2f} GB")
    print(f"[AWQ] 保存至: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="模型量化 (GPTQ / AWQ)")
    parser.add_argument(
        "--model-path",
        type=str,
        default="/root/autodl-tmp/ustc-qa-merged",
        help="合并后模型的路径",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="outputs/ustc-qa-quantized",
        help="量化后模型的输出路径",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="awq",
        choices=["gptq", "awq"],
        help="量化方法: gptq 或 awq (推荐 awq)",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=4,
        choices=[4, 8],
        help="量化位数",
    )
    args = parser.parse_args()

    # 自动追加量化方法到输出路径
    output_path = f"{args.output_path}-{args.method}-int{args.bits}"

    print("=" * 50)
    print(f"  模型量化")
    print(f"  方法: {args.method.upper()}")
    print(f"  位数: INT{args.bits}")
    print(f"  输入: {args.model_path}")
    print(f"  输出: {output_path}")
    print("=" * 50)

    if args.method == "gptq":
        quantize_gptq(args.model_path, output_path, args.bits)
    else:
        quantize_awq(args.model_path, output_path, args.bits)

    print(f"\n下一步: 运行 bash deploy/serve.sh --model {output_path} 启动推理服务")


if __name__ == "__main__":
    main()
