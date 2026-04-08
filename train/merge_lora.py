"""
LoRA 权重合并脚本
将 LoRA adapter 合并到基座模型中，导出完整模型用于后续量化和部署。
"""

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge_lora_weights(
    base_model_path: str,
    lora_adapter_path: str,
    output_path: str,
    torch_dtype: str = "bfloat16",
):
    """合并 LoRA 权重到基座模型"""

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(torch_dtype, torch.bfloat16)

    print("=" * 50)
    print("  LoRA 权重合并")
    print(f"  基座模型: {base_model_path}")
    print(f"  LoRA 适配器: {lora_adapter_path}")
    print(f"  输出路径: {output_path}")
    print(f"  精度: {torch_dtype}")
    print("=" * 50)

    # Step 1: 加载 tokenizer
    print("\n[1/4] 加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path, trust_remote_code=True
    )

    # Step 2: 加载基座模型
    print("[2/4] 加载基座模型...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=dtype,
        device_map="cpu",  # 在 CPU 上合并，避免显存不足
        trust_remote_code=True,
    )

    # Step 3: 加载并合并 LoRA
    print("[3/4] 加载 LoRA 适配器并合并...")
    model = PeftModel.from_pretrained(base_model, lora_adapter_path)
    model = model.merge_and_unload()

    # Step 4: 保存合并后的模型
    print("[4/4] 保存合并后的模型...")
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    # 计算模型大小
    total_size = sum(
        f.stat().st_size for f in output_dir.glob("*.safetensors")
    )
    print(f"\n✅ 合并完成!")
    print(f"   模型大小: {total_size / 1024**3:.2f} GB")
    print(f"   保存至: {output_dir}")
    print(f"   下一步: 运行 python quantize/quantize_model.py 进行量化")


def main():
    parser = argparse.ArgumentParser(description="合并 LoRA 权重到基座模型")
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen2.5-7B",
        help="基座模型路径或 HuggingFace 模型名",
    )
    parser.add_argument(
        "--lora-adapter",
        type=str,
        default="outputs/ustc-qa-lora",
        help="LoRA 适配器路径",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/ustc-qa-merged",
        help="合并后模型的输出路径",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="模型精度",
    )
    args = parser.parse_args()

    merge_lora_weights(
        base_model_path=args.base_model,
        lora_adapter_path=args.lora_adapter,
        output_path=args.output,
        torch_dtype=args.dtype,
    )


if __name__ == "__main__":
    main()
