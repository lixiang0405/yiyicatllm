"""
模型量化脚本
使用 transformers 原生 GPTQConfig 进行 GPTQ 量化，无需额外安装 auto-gptq / autoawq。
将 FP16/BF16 模型压缩为 INT4，适配 8GB 显存部署。

依赖：transformers, optimum, torch
"""

# 绕过 torchvision CUDA 版本检查：量化语言模型不需要 torchvision，
# 但 transformers 的 import 链条会拖进来导致 CUDA 版本不匹配报错
import importlib
import importlib.util
import sys
import types

if "torchvision" not in sys.modules:
    fake_tv = types.ModuleType("torchvision")
    fake_tv.__version__ = "0.25.0"
    fake_tv.__spec__ = importlib.util.spec_from_loader("torchvision", loader=None)
    fake_transforms = types.ModuleType("torchvision.transforms")
    fake_transforms.__spec__ = importlib.util.spec_from_loader("torchvision.transforms", loader=None)
    fake_transforms.InterpolationMode = type("InterpolationMode", (), {
        "BILINEAR": 2, "BICUBIC": 3, "NEAREST": 0,
    })
    fake_tv.transforms = fake_transforms
    sys.modules["torchvision"] = fake_tv
    sys.modules["torchvision.transforms"] = fake_transforms

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig


def load_calibration_texts(data_dir: str = "data", max_texts: int = 128):
    """从训练数据中加载校准文本，确保校准数据与训练分布一致"""
    import random
    data_path = Path(data_dir)
    texts = []

    # 从 new_qa.json 加载（instruction + output 拼接成完整对话）
    new_qa_path = data_path / "new_qa.json"
    if new_qa_path.exists():
        with open(new_qa_path, "r", encoding="utf-8") as f:
            for item in json.load(f):
                question = item.get("instruction", "")
                answer = item.get("output", "")
                if question and answer:
                    texts.append(f"问：{question}\n答：{answer}")

    # 从 preference_data.json 加载（instruction + chosen）
    pref_path = data_path / "preference_data.json"
    if pref_path.exists():
        with open(pref_path, "r", encoding="utf-8") as f:
            for item in json.load(f):
                question = item.get("instruction", "")
                answer = item.get("chosen", "")
                if question and answer:
                    texts.append(f"问：{question}\n答：{answer}")

    if not texts:
        raise FileNotFoundError(
            f"在 {data_dir} 下未找到 new_qa.json 或 preference_data.json，"
            "请确保训练数据文件存在"
        )

    # 去重后随机采样
    texts = list(set(texts))
    random.seed(42)
    random.shuffle(texts)
    sampled = texts[:max_texts]
    print(f"[校准数据] 共加载 {len(texts)} 条，采样 {len(sampled)} 条")
    return sampled


def prepare_calibration_dataset(tokenizer, data_dir: str = "data",
                                num_samples: int = 128, max_length: int = 512):
    """准备校准数据集，从真实训练数据中采样"""
    calibration_texts = load_calibration_texts(data_dir, max_texts=num_samples)
    return [
        tokenizer(text, return_tensors="pt", padding="max_length",
                  truncation=True, max_length=max_length)["input_ids"].squeeze(0)
        for text in calibration_texts
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