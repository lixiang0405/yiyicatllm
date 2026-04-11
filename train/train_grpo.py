"""
自实现在线 GRPO (Group Relative Policy Optimization) 训练
绕开 veRL hybrid engine 的 vLLM v1 兼容性问题

核心思路:
  训练和推理交替进行，不同时占用 GPU 显存：
  1. 用 vLLM 离线批量生成 n 个回答 (占满 GPU)
  2. 释放 vLLM 显存
  3. 用 PyTorch + PEFT LoRA 做 GRPO 策略更新 (占满 GPU)
  4. 合并 LoRA 权重到基础模型
  5. 重复 1-4

GRPO 算法 (DeepSeek-R1):
  - 对每个 prompt 采样 n 个回答
  - 用规则奖励函数打分
  - 用组内 reward 的均值作为 baseline (不需要 Critic)
  - advantage = (reward - mean) / std
  - PPO-clip 目标函数更新策略

用法:
  python train/train_grpo.py \
      --model /root/autodl-tmp/ustc-qa-dpo-merged \
      --data data/grpo_prompts.parquet \
      --output outputs/ustc-qa-grpo
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
# vLLM v1 CuMemAllocator 不兼容 expandable_segments
if "PYTORCH_CUDA_ALLOC_CONF" in os.environ:
    del os.environ["PYTORCH_CUDA_ALLOC_CONF"]

import gc
import sys
import json
import time
import math
import argparse
import subprocess
from pathlib import Path
from typing import List, Tuple

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

# 添加项目根目录到 path，以便导入 reward_function
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "train"))
from reward_function import compute_reward


# ============================================
# 工具函数
# ============================================

def log(message: str):
    """带时间戳的日志"""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def get_gpu_memory_info() -> str:
    """获取 GPU 显存使用情况"""
    if not torch.cuda.is_available():
        return "No GPU"
    info_parts = []
    for i in range(torch.cuda.device_count()):
        used = torch.cuda.memory_allocated(i) / 1024**3
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        info_parts.append(f"GPU{i}: {used:.1f}/{total:.1f}GB")
    return " | ".join(info_parts)


def free_gpu_memory():
    """彻底释放 GPU 显存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def load_prompts(data_path: str) -> List[str]:
    """从 parquet 文件加载 prompt（提取 user content）"""
    dataframe = pd.read_parquet(data_path)
    prompts = []
    for raw_prompt in dataframe["prompt"]:
        messages = json.loads(raw_prompt)
        user_content = messages[0]["content"]
        prompts.append(user_content)
    return prompts


# ============================================
# Phase 1: vLLM 批量生成
# ============================================

def generate_responses_vllm(
    model_path: str,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    num_samples: int = 4,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    tensor_parallel_size: int = None,
) -> Tuple[List[List[str]], List[List[List[int]]]]:
    """
    用 vLLM 离线批量生成回答

    Args:
        model_path: 模型路径
        tokenizer: tokenizer（用于构建 chat prompt）
        prompts: 原始问题列表
        num_samples: 每个 prompt 采样数
        max_new_tokens: 最大生成长度
        temperature: 采样温度
        top_p: nucleus sampling
        tensor_parallel_size: TP 并行度

    Returns:
        all_responses: [num_prompts][num_samples] 的回答文本
        all_token_ids: [num_prompts][num_samples] 的 token id 列表
    """
    from vllm import LLM, SamplingParams

    if tensor_parallel_size is None:
        tensor_parallel_size = torch.cuda.device_count()

    log(f"加载 vLLM 模型: {model_path} (TP={tensor_parallel_size})")
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        tensor_parallel_size=tensor_parallel_size,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        enforce_eager=True,
    )

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        n=num_samples,
        stop=["<|im_end|>", "<|endoftext|>", "<|im_start|>"],
    )

    # 构建 chat 格式 prompt
    system_prompt = (
        "你是中国科学技术大学软件学院的智能问答助手。"
        "请根据你的知识详细回答用户的问题，尽量提供具体的信息。"
    )
    formatted_prompts = []
    for question in prompts:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        formatted_prompts.append(formatted)

    log(f"开始生成: {len(formatted_prompts)} 个 prompt × {num_samples} 个采样")
    start_time = time.time()
    outputs = llm.generate(formatted_prompts, sampling_params)
    elapsed = time.time() - start_time
    log(f"生成完成: {elapsed:.1f}s ({len(formatted_prompts) * num_samples / elapsed:.1f} 条/秒)")

    # 解析结果
    all_responses = []
    all_token_ids = []
    for output in outputs:
        responses = []
        token_ids = []
        for sample in output.outputs:
            responses.append(sample.text.strip())
            token_ids.append(list(sample.token_ids))
        all_responses.append(responses)
        all_token_ids.append(token_ids)

    # 释放 vLLM 显存
    log("释放 vLLM 显存...")
    del llm
    free_gpu_memory()
    log(f"显存释放完成: {get_gpu_memory_info()}")

    return all_responses, all_token_ids


# ============================================
# Phase 2: 批量计算 log_prob（left-padding + attention_mask）
# ============================================

SYSTEM_PROMPT = (
    "你是中国科学技术大学软件学院的智能问答助手。"
    "请根据你的知识详细回答用户的问题，尽量提供具体的信息。"
)


def _prepare_sequences(
    tokenizer: AutoTokenizer,
    prompts: List[str],
    responses: List[str],
    max_length: int = 512,
) -> Tuple[List[List[int]], List[int]]:
    """
    将 (prompt, response) 对编码为 token id 序列，并记录 response 起始位置

    Returns:
        all_full_ids: 每条的完整 token id 列表
        all_response_starts: 每条的 response 起始位置
    """
    all_full_ids = []
    all_response_starts = []

    for prompt_text, response_text in zip(prompts, responses):
        full_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text},
            {"role": "assistant", "content": response_text},
        ]
        full_text = tokenizer.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )

        prompt_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text},
        ]
        prompt_text_only = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )

        full_ids = tokenizer.encode(full_text, add_special_tokens=False,
                                     max_length=max_length, truncation=True)
        prompt_ids = tokenizer.encode(prompt_text_only, add_special_tokens=False,
                                       max_length=max_length, truncation=True)

        all_full_ids.append(full_ids)
        all_response_starts.append(len(prompt_ids))

    return all_full_ids, all_response_starts


def _pad_and_batch(
    all_full_ids: List[List[int]],
    pad_token_id: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Left-padding 并构建 attention_mask

    Returns:
        input_ids: [batch, max_len]
        attention_mask: [batch, max_len]
    """
    max_len = max(len(ids) for ids in all_full_ids)
    padded_ids = []
    masks = []
    for ids in all_full_ids:
        pad_len = max_len - len(ids)
        padded_ids.append([pad_token_id] * pad_len + ids)
        masks.append([0] * pad_len + [1] * len(ids))

    return (
        torch.tensor(padded_ids, device=device),
        torch.tensor(masks, device=device),
    )


def _extract_response_log_probs(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    all_full_ids: List[List[int]],
    all_response_starts: List[int],
    max_len: int,
) -> torch.Tensor:
    """
    从 batch logits 中提取每条 response 部分的 log_prob 之和

    Args:
        logits: [batch, max_len, vocab_size]
        input_ids: [batch, max_len]
        all_full_ids: 原始未 padding 的 token id
        all_response_starts: response 起始位置（在原始序列中）
        max_len: padding 后的最大长度

    Returns:
        log_probs: [batch] 的 log_prob 张量
    """
    log_probs_all = torch.nn.functional.log_softmax(logits, dim=-1)
    batch_log_probs = []

    for i in range(len(all_full_ids)):
        seq_len = len(all_full_ids[i])
        pad_len = max_len - seq_len
        response_start = all_response_starts[i]

        if response_start >= seq_len:
            batch_log_probs.append(logits.new_tensor(-100.0))
            continue

        # 在 padded 序列中的绝对位置
        abs_start = pad_len + response_start
        abs_end = pad_len + seq_len

        # logits[t] 预测 token[t+1]
        resp_logits = log_probs_all[i, abs_start - 1: abs_end - 1, :]  # [resp_len, vocab]
        resp_token_ids = input_ids[i, abs_start: abs_end]  # [resp_len]

        token_lp = resp_logits.gather(1, resp_token_ids.unsqueeze(1)).squeeze(1)
        batch_log_probs.append(token_lp.sum())

    return torch.stack(batch_log_probs)


def compute_log_probs_batched(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    responses: List[str],
    device: torch.device,
    max_length: int = 512,
    micro_batch_size: int = 8,
    requires_grad: bool = False,
) -> torch.Tensor:
    """
    批量计算 log_prob（支持有/无梯度模式）

    Args:
        model: HF 模型
        tokenizer: tokenizer
        prompts: 问题列表
        responses: 回答列表
        device: 计算设备
        max_length: 最大序列长度
        micro_batch_size: 每个 micro batch 的大小
        requires_grad: 是否保留梯度

    Returns:
        log_probs: [total_size] 的 log_prob 张量
    """
    all_full_ids, all_response_starts = _prepare_sequences(
        tokenizer, prompts, responses, max_length
    )

    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    all_log_probs = []

    for mb_start in range(0, len(prompts), micro_batch_size):
        mb_end = min(mb_start + micro_batch_size, len(prompts))
        mb_ids = all_full_ids[mb_start:mb_end]
        mb_starts = all_response_starts[mb_start:mb_end]

        input_ids, attention_mask = _pad_and_batch(mb_ids, pad_token_id, device)
        max_len = input_ids.shape[1]

        if requires_grad:
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        else:
            with torch.no_grad():
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

        mb_log_probs = _extract_response_log_probs(
            logits, input_ids, mb_ids, mb_starts, max_len
        )
        all_log_probs.append(mb_log_probs)

        # 及时释放中间张量
        del logits, input_ids, attention_mask

    return torch.cat(all_log_probs)


# ============================================
# Phase 3: GRPO 训练
# ============================================

def grpo_train_step(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    optimizer: torch.optim.Optimizer,
    flat_prompts: List[str],
    flat_responses: List[str],
    advantages: torch.Tensor,
    ref_log_probs: torch.Tensor,
    device: torch.device,
    clip_epsilon: float = 0.2,
    kl_coef: float = 0.002,
    max_length: int = 512,
    micro_batch_size: int = 8,
) -> dict:
    """
    执行一个 GRPO 训练 step（已展平的数据）

    Args:
        model: 策略模型 (带 LoRA)
        tokenizer: tokenizer
        optimizer: 优化器
        flat_prompts: 展平的问题列表 [batch_size * num_samples]
        flat_responses: 展平的回答列表
        advantages: 归一化后的 advantage [batch_size * num_samples]
        ref_log_probs: 参考模型的 log_prob [batch_size * num_samples]
        device: 计算设备
        clip_epsilon: PPO clip 范围
        kl_coef: KL 散度惩罚系数
        max_length: 最大序列长度
        micro_batch_size: micro batch 大小

    Returns:
        metrics: 训练指标字典
    """
    model.train()
    total_samples = len(flat_prompts)
    num_micro_batches = math.ceil(total_samples / micro_batch_size)

    total_loss = torch.tensor(0.0, device=device)
    total_policy_loss = torch.tensor(0.0, device=device)
    total_kl = torch.tensor(0.0, device=device)

    for mb_start in range(0, total_samples, micro_batch_size):
        mb_end = min(mb_start + micro_batch_size, total_samples)
        mb_prompts = flat_prompts[mb_start:mb_end]
        mb_responses = flat_responses[mb_start:mb_end]
        mb_advantages = advantages[mb_start:mb_end]
        mb_ref_lp = ref_log_probs[mb_start:mb_end]

        # 当前策略的 log_prob（保留梯度）
        current_lp = compute_log_probs_batched(
            model, tokenizer, mb_prompts, mb_responses,
            device, max_length, micro_batch_size=len(mb_prompts),
            requires_grad=True,
        )

        # PPO-clip 目标（clamp log_ratio 防止 exp 爆炸）
        log_ratio = current_lp - mb_ref_lp.detach()
        log_ratio = torch.clamp(log_ratio, -5.0, 5.0)  # 防止 ratio 爆炸
        ratio = torch.exp(log_ratio)
        surr1 = ratio * mb_advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * mb_advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # KL 散度惩罚
        kl_divergence = (mb_ref_lp.detach() - current_lp).mean()
        kl_loss = kl_coef * kl_divergence

        loss = (policy_loss + kl_loss) / num_micro_batches
        loss.backward()

        total_loss += loss.detach()
        total_policy_loss += policy_loss.detach()
        total_kl += kl_divergence.detach()

    # 梯度更新
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad()

    metrics = {
        "loss": total_loss.item(),
        "policy_loss": (total_policy_loss / num_micro_batches).item(),
        "kl_divergence": (total_kl / num_micro_batches).item(),
    }
    return metrics


# ============================================
# Phase 4: LoRA 合并与保存
# ============================================

def merge_and_save_lora(model, tokenizer, output_path: str):
    """合并 LoRA 权重并保存完整模型"""
    log(f"合并 LoRA 权重到: {output_path}")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    del merged_model
    free_gpu_memory()
    log("LoRA 合并保存完成")


# ============================================
# 主训练循环
# ============================================

def main():
    parser = argparse.ArgumentParser(description="自实现在线 GRPO 训练")
    parser.add_argument("--model", type=str, default="/root/autodl-tmp/ustc-qa-dpo-merged",
                        help="基础模型路径 (DPO 合并后)")
    parser.add_argument("--data", type=str, default="data/grpo_prompts.parquet",
                        help="GRPO 训练数据路径")
    parser.add_argument("--output", type=str, default="outputs/ustc-qa-grpo",
                        help="输出目录")
    parser.add_argument("--num-samples", type=int, default=4,
                        help="每个 prompt 采样数 (GRPO group size)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="每个训练 step 的 prompt 数")
    parser.add_argument("--epochs", type=int, default=1,
                        help="训练轮数")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="学习率")
    parser.add_argument("--lora-rank", type=int, default=32,
                        help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=64,
                        help="LoRA alpha")
    parser.add_argument("--clip-epsilon", type=float, default=0.2,
                        help="PPO clip epsilon")
    parser.add_argument("--kl-coef", type=float, default=0.002,
                        help="KL 散度惩罚系数")
    parser.add_argument("--max-length", type=int, default=512,
                        help="最大序列长度")
    parser.add_argument("--max-new-tokens", type=int, default=256,
                        help="最大生成长度")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="采样温度")
    parser.add_argument("--save-steps", type=int, default=50,
                        help="每多少步保存一次 checkpoint")
    parser.add_argument("--skip-generate", action="store_true",
                        help="跳过推理，复用上次缓存的生成数据")
    args = parser.parse_args()

    print("=" * 60)
    print("  中科大智能问答助手 - 自实现在线 GRPO 训练")
    print("=" * 60)
    print(f"  模型: {args.model}")
    print(f"  数据: {args.data}")
    print(f"  输出: {args.output}")
    print(f"  GPU: {torch.cuda.device_count()} × {torch.cuda.get_device_name(0)}")
    print(f"  每 prompt 采样: {args.num_samples}")
    print(f"  训练 batch: {args.batch_size}")
    print(f"  LoRA: rank={args.lora_rank}, alpha={args.lora_alpha}")
    print(f"  学习率: {args.lr}")
    print(f"  PPO clip: {args.clip_epsilon}")
    print(f"  KL coef: {args.kl_coef}")
    print("=" * 60)

    # 加载数据
    log("加载训练数据...")
    all_prompts = load_prompts(args.data)
    log(f"共 {len(all_prompts)} 条 prompt")

    # 加载 tokenizer（全程共用）
    log("加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 当前模型路径（每轮生成后会更新）
    current_model_path = args.model

    for epoch in range(args.epochs):
        log(f"\n{'='*60}")
        log(f"  Epoch {epoch + 1}/{args.epochs}")
        log(f"{'='*60}")

        # ========================================
        # Phase 1: vLLM 一次性批量生成所有回答（支持缓存）
        # ========================================
        cache_file = output_dir / f"generate_cache_epoch{epoch+1}.json"

        if args.skip_generate and cache_file.exists():
            log(f"\n[Phase 1] 跳过推理，加载缓存: {cache_file}")
            with open(cache_file, "r", encoding="utf-8") as f:
                cache_data = json.load(f)
            all_responses = cache_data["responses"]
            all_token_ids = cache_data.get("token_ids", [[] for _ in all_responses])
            log(f"  加载完成: {len(all_responses)} 条 prompt 的生成结果")
        else:
            log("\n[Phase 1] vLLM 批量生成回答...")
            log(f"  共 {len(all_prompts)} 条 prompt × {args.num_samples} 采样")

            all_responses, all_token_ids = generate_responses_vllm(
                model_path=current_model_path,
                tokenizer=tokenizer,
                prompts=all_prompts,
                num_samples=args.num_samples,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )

            # 缓存生成结果
            log(f"  缓存生成结果到: {cache_file}")
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump({"responses": all_responses}, f, ensure_ascii=False)

        # 计算所有 reward
        log("\n[Phase 1.5] 计算奖励...")
        all_rewards = []
        for prompt_text, responses in zip(all_prompts, all_responses):
            prompt_list = [prompt_text] * len(responses)
            rewards = compute_reward(prompt_list, responses)
            all_rewards.append(rewards)

        flat_rewards = [r for group in all_rewards for r in group]
        log(f"  平均奖励: {sum(flat_rewards)/len(flat_rewards):.3f}")
        log(f"  最高奖励: {max(flat_rewards):.3f}")
        log(f"  最低奖励: {min(flat_rewards):.3f}")

        # ========================================
        # Phase 2: 展平数据 + 计算 advantage
        # ========================================
        log("\n[Phase 2] 计算 GRPO advantage...")
        all_flat_prompts = []
        all_flat_responses = []
        all_advantages = []
        all_flat_rewards_list = []

        for prompt_text, responses, rewards in zip(all_prompts, all_responses, all_rewards):
            reward_tensor = torch.tensor(rewards, dtype=torch.float32)
            mean_reward = reward_tensor.mean()
            std_reward = reward_tensor.std()
            if std_reward < 1e-8:
                std_reward = torch.tensor(1.0)
            advantages = (reward_tensor - mean_reward) / std_reward

            for response_text, advantage, reward_val in zip(responses, advantages, rewards):
                all_flat_prompts.append(prompt_text)
                all_flat_responses.append(response_text)
                all_advantages.append(advantage)
                all_flat_rewards_list.append(reward_val)

        device = torch.device("cuda:0")
        advantages_tensor = torch.stack(all_advantages).to(device)
        log(f"  展平后: {len(all_flat_prompts)} 条 (prompt, response) 对")

        # ========================================
        # Phase 2.5: 加载 ref_model，计算所有 ref log_prob，然后释放
        # ========================================
        log("\n[Phase 2.5] 计算参考模型 log_prob...")
        free_gpu_memory()

        ref_model = AutoModelForCausalLM.from_pretrained(
            current_model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True,
        )
        ref_model.eval()
        log(f"  参考模型加载完成: {get_gpu_memory_info()}")

        ref_log_probs = compute_log_probs_batched(
            ref_model, tokenizer, all_flat_prompts, all_flat_responses,
            device, args.max_length, micro_batch_size=16, requires_grad=False,
        )
        log(f"  ref log_prob 计算完成: shape={ref_log_probs.shape}")

        # 释放 ref_model，腾出显存给 policy_model
        del ref_model
        free_gpu_memory()
        log(f"  参考模型已释放: {get_gpu_memory_info()}")

        # ========================================
        # Phase 3: 加载 policy_model + LoRA 训练
        # ========================================
        log("\n[Phase 3] 加载策略模型 + LoRA...")
        from peft import LoraConfig, get_peft_model

        policy_model = AutoModelForCausalLM.from_pretrained(
            current_model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True,
        )

        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules="all-linear",
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        policy_model = get_peft_model(policy_model, lora_config)
        trainable_params = sum(p.numel() for p in policy_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in policy_model.parameters())
        log(f"  可训练参数: {trainable_params/1e6:.1f}M / {total_params/1e6:.1f}M "
            f"({trainable_params/total_params*100:.2f}%)")
        log(f"  显存: {get_gpu_memory_info()}")

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, policy_model.parameters()),
            lr=args.lr,
            weight_decay=0.01,
        )

        # GRPO 训练循环
        log("\n  开始 GRPO 训练...")
        num_samples_per_step = args.batch_size * args.num_samples  # 展平后每 step 的样本数
        num_steps = len(all_flat_prompts) // num_samples_per_step
        if num_steps == 0:
            num_steps = 1
            num_samples_per_step = len(all_flat_prompts)
        log(f"  总步数: {num_steps} (每步 {num_samples_per_step} 条)")

        epoch_metrics = {"loss": 0, "policy_loss": 0, "kl_divergence": 0}
        step_count = 0

        for step in range(num_steps):
            batch_start = step * num_samples_per_step
            batch_end = min(batch_start + num_samples_per_step, len(all_flat_prompts))

            metrics = grpo_train_step(
                model=policy_model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                flat_prompts=all_flat_prompts[batch_start:batch_end],
                flat_responses=all_flat_responses[batch_start:batch_end],
                advantages=advantages_tensor[batch_start:batch_end],
                ref_log_probs=ref_log_probs[batch_start:batch_end],
                device=device,
                clip_epsilon=args.clip_epsilon,
                kl_coef=args.kl_coef,
                max_length=args.max_length,
                micro_batch_size=8,
            )

            step_count += 1
            for key in epoch_metrics:
                if key in metrics:
                    epoch_metrics[key] += metrics[key]

            # 计算当前 batch 的平均 reward
            batch_rewards = all_flat_rewards_list[batch_start:batch_end]
            mean_reward = sum(batch_rewards) / len(batch_rewards)

            if (step + 1) % 5 == 0 or step == 0:
                log(f"  Step {step+1}/{num_steps} | "
                    f"loss={metrics['loss']:.4f} | "
                    f"policy={metrics['policy_loss']:.4f} | "
                    f"kl={metrics['kl_divergence']:.4f} | "
                    f"reward={mean_reward:.3f}")

            # 定期保存 checkpoint
            if (step + 1) % args.save_steps == 0:
                ckpt_path = output_dir / f"checkpoint-epoch{epoch+1}-step{step+1}"
                policy_model.save_pretrained(str(ckpt_path))
                tokenizer.save_pretrained(str(ckpt_path))
                log(f"  Checkpoint 保存: {ckpt_path}")

        # Epoch 结束，打印平均指标
        if step_count > 0:
            for key in epoch_metrics:
                epoch_metrics[key] /= step_count
        avg_reward = sum(all_flat_rewards_list) / len(all_flat_rewards_list)
        log(f"\n  Epoch {epoch+1} 平均指标:")
        log(f"    loss={epoch_metrics['loss']:.4f}")
        log(f"    policy_loss={epoch_metrics['policy_loss']:.4f}")
        log(f"    kl_divergence={epoch_metrics['kl_divergence']:.4f}")
        log(f"    mean_reward={avg_reward:.3f}")

        # ========================================
        # Phase 4: 合并 LoRA 并保存
        # ========================================
        log("\n[Phase 4] 合并 LoRA 权重...")
        epoch_output = str(output_dir / f"epoch-{epoch+1}")
        merge_and_save_lora(policy_model, tokenizer, epoch_output)

        # 释放训练模型显存
        del policy_model, optimizer, ref_log_probs, advantages_tensor
        free_gpu_memory()

        # 更新模型路径，下一轮用合并后的模型
        current_model_path = epoch_output
        log(f"  下一轮将使用: {current_model_path}")

    # ========================================
    # 训练完成
    # ========================================
    # 复制最终模型到输出根目录
    final_model_path = str(output_dir / "final")
    log(f"\n复制最终模型到: {final_model_path}")
    import shutil
    if Path(current_model_path).exists():
        shutil.copytree(current_model_path, final_model_path, dirs_exist_ok=True)

    print("\n" + "=" * 60)
    print("  GRPO 训练完成!")
    print(f"  最终模型: {final_model_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
