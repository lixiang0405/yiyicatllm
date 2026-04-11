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
        total = torch.cuda.get_device_properties(i).total_mem / 1024**3
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
# Phase 2: 计算 log_prob（用 HF 模型）
# ============================================

def compute_log_probs_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    responses: List[str],
    device: torch.device,
    max_length: int = 512,
) -> torch.Tensor:
    """
    计算一批 (prompt, response) 对的 response 部分 log_prob 之和

    Args:
        model: HF 模型
        tokenizer: tokenizer
        prompts: 问题列表
        responses: 回答列表
        device: 计算设备
        max_length: 最大序列长度

    Returns:
        log_probs: [batch_size] 的 log_prob 张量
    """
    system_prompt = (
        "你是中国科学技术大学软件学院的智能问答助手。"
        "请根据你的知识详细回答用户的问题，尽量提供具体的信息。"
    )

    all_log_probs = []

    for prompt_text, response_text in zip(prompts, responses):
        # 构建完整的 chat 序列
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_text},
            {"role": "assistant", "content": response_text},
        ]
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        # 构建只到 assistant 开头的 prompt 部分，用于确定 response 起始位置
        prompt_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_text},
        ]
        prompt_text_only = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )

        # tokenize
        full_ids = tokenizer.encode(full_text, add_special_tokens=False, max_length=max_length, truncation=True)
        prompt_ids = tokenizer.encode(prompt_text_only, add_special_tokens=False, max_length=max_length, truncation=True)

        response_start = len(prompt_ids)

        if response_start >= len(full_ids):
            all_log_probs.append(torch.tensor(-100.0, device=device))
            continue

        input_ids = torch.tensor([full_ids], device=device)

        with torch.no_grad():
            logits = model(input_ids).logits  # [1, seq_len, vocab_size]

        # 计算 response 部分的 log_prob
        # logits[t] 预测 token[t+1]，所以 response token 从 response_start 开始
        # 对应的 logits 从 response_start-1 开始
        response_logits = logits[0, response_start - 1: len(full_ids) - 1, :]  # [resp_len, vocab]
        response_token_ids = torch.tensor(full_ids[response_start:], device=device)  # [resp_len]

        log_probs_all = torch.nn.functional.log_softmax(response_logits, dim=-1)
        token_log_probs = log_probs_all.gather(1, response_token_ids.unsqueeze(1)).squeeze(1)
        total_log_prob = token_log_probs.sum()

        all_log_probs.append(total_log_prob)

    return torch.stack(all_log_probs)


# ============================================
# Phase 3: GRPO 训练
# ============================================

def grpo_train_step(
    model: AutoModelForCausalLM,
    ref_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    optimizer: torch.optim.Optimizer,
    prompts: List[str],
    grouped_responses: List[List[str]],
    grouped_rewards: List[List[float]],
    device: torch.device,
    clip_epsilon: float = 0.2,
    kl_coef: float = 0.002,
    max_length: int = 512,
) -> dict:
    """
    执行一个 GRPO 训练 step

    Args:
        model: 策略模型 (带 LoRA)
        ref_model: 参考模型 (冻结)
        tokenizer: tokenizer
        optimizer: 优化器
        prompts: 问题列表 [batch_size]
        grouped_responses: 回答列表 [batch_size][num_samples]
        grouped_rewards: 奖励列表 [batch_size][num_samples]
        device: 计算设备
        clip_epsilon: PPO clip 范围
        kl_coef: KL 散度惩罚系数
        max_length: 最大序列长度

    Returns:
        metrics: 训练指标字典
    """
    model.train()
    num_samples = len(grouped_responses[0])

    # 计算 GRPO advantage: (reward - group_mean) / group_std
    all_advantages = []
    all_flat_prompts = []
    all_flat_responses = []

    for prompt_text, responses, rewards in zip(prompts, grouped_responses, grouped_rewards):
        reward_tensor = torch.tensor(rewards, dtype=torch.float32)
        mean_reward = reward_tensor.mean()
        std_reward = reward_tensor.std()
        if std_reward < 1e-8:
            std_reward = torch.tensor(1.0)
        advantages = (reward_tensor - mean_reward) / std_reward

        for response_text, advantage in zip(responses, advantages):
            all_flat_prompts.append(prompt_text)
            all_flat_responses.append(response_text)
            all_advantages.append(advantage)

    advantages_tensor = torch.stack(all_advantages).to(device)

    # 计算 ref model 的 log_prob（不需要梯度）
    ref_model.eval()
    with torch.no_grad():
        ref_log_probs = compute_log_probs_batch(
            ref_model, tokenizer, all_flat_prompts, all_flat_responses, device, max_length
        )

    # 计算当前策略的 log_prob（需要梯度）
    model.train()
    # 分 micro batch 计算，避免 OOM
    micro_batch_size = 4
    total_loss = torch.tensor(0.0, device=device)
    total_policy_loss = torch.tensor(0.0, device=device)
    total_kl = torch.tensor(0.0, device=device)
    num_micro_batches = 0

    for mb_start in range(0, len(all_flat_prompts), micro_batch_size):
        mb_end = min(mb_start + micro_batch_size, len(all_flat_prompts))
        mb_prompts = all_flat_prompts[mb_start:mb_end]
        mb_responses = all_flat_responses[mb_start:mb_end]
        mb_advantages = advantages_tensor[mb_start:mb_end]
        mb_ref_log_probs = ref_log_probs[mb_start:mb_end]

        # 当前策略的 log_prob
        current_log_probs = compute_log_probs_with_grad(
            model, tokenizer, mb_prompts, mb_responses, device, max_length
        )

        # log ratio = log(pi/pi_old)，这里用 ref 作为 pi_old
        log_ratio = current_log_probs - mb_ref_log_probs.detach()
        ratio = torch.exp(log_ratio)

        # PPO-clip 目标
        surr1 = ratio * mb_advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * mb_advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # KL 散度惩罚 (low variance KL estimator)
        kl_divergence = (mb_ref_log_probs.detach() - current_log_probs).mean()
        kl_loss = kl_coef * kl_divergence

        loss = (policy_loss + kl_loss) / math.ceil(len(all_flat_prompts) / micro_batch_size)
        loss.backward()

        total_loss += loss.detach()
        total_policy_loss += policy_loss.detach()
        total_kl += kl_divergence.detach()
        num_micro_batches += 1

    # 梯度更新
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad()

    # 汇总指标
    flat_rewards = [r for group in grouped_rewards for r in group]
    metrics = {
        "loss": total_loss.item(),
        "policy_loss": (total_policy_loss / num_micro_batches).item(),
        "kl_divergence": (total_kl / num_micro_batches).item(),
        "mean_reward": sum(flat_rewards) / len(flat_rewards),
        "max_reward": max(flat_rewards),
        "min_reward": min(flat_rewards),
    }
    return metrics


def compute_log_probs_with_grad(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    responses: List[str],
    device: torch.device,
    max_length: int = 512,
) -> torch.Tensor:
    """计算 log_prob（保留梯度），逐条处理避免 padding 复杂度"""
    system_prompt = (
        "你是中国科学技术大学软件学院的智能问答助手。"
        "请根据你的知识详细回答用户的问题，尽量提供具体的信息。"
    )

    all_log_probs = []

    for prompt_text, response_text in zip(prompts, responses):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_text},
            {"role": "assistant", "content": response_text},
        ]
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        prompt_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_text},
        ]
        prompt_text_only = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )

        full_ids = tokenizer.encode(full_text, add_special_tokens=False, max_length=max_length, truncation=True)
        prompt_ids = tokenizer.encode(prompt_text_only, add_special_tokens=False, max_length=max_length, truncation=True)

        response_start = len(prompt_ids)

        if response_start >= len(full_ids):
            all_log_probs.append(torch.tensor(-100.0, device=device, requires_grad=True))
            continue

        input_ids = torch.tensor([full_ids], device=device)
        logits = model(input_ids).logits

        response_logits = logits[0, response_start - 1: len(full_ids) - 1, :]
        response_token_ids = torch.tensor(full_ids[response_start:], device=device)

        log_probs_all = torch.nn.functional.log_softmax(response_logits, dim=-1)
        token_log_probs = log_probs_all.gather(1, response_token_ids.unsqueeze(1)).squeeze(1)
        total_log_prob = token_log_probs.sum()

        all_log_probs.append(total_log_prob)

    return torch.stack(all_log_probs)


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
    parser.add_argument("--generate-batch-size", type=int, default=256,
                        help="vLLM 一次生成的 prompt 数量")
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
        # Phase 1: vLLM 批量生成所有回答
        # ========================================
        log("\n[Phase 1] vLLM 批量生成回答...")

        # 分批生成（vLLM 内部会高效处理）
        all_responses = []
        all_token_ids = []

        for gen_start in range(0, len(all_prompts), args.generate_batch_size):
            gen_end = min(gen_start + args.generate_batch_size, len(all_prompts))
            batch_prompts = all_prompts[gen_start:gen_end]

            log(f"  生成批次 {gen_start//args.generate_batch_size + 1}: "
                f"prompt {gen_start+1}-{gen_end}/{len(all_prompts)}")

            batch_responses, batch_token_ids = generate_responses_vllm(
                model_path=current_model_path,
                tokenizer=tokenizer,
                prompts=batch_prompts,
                num_samples=args.num_samples,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            all_responses.extend(batch_responses)
            all_token_ids.extend(batch_token_ids)

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
        # Phase 2: 加载模型 + LoRA 训练
        # ========================================
        log("\n[Phase 2] 加载训练模型...")
        free_gpu_memory()

        device = torch.device("cuda:0")

        # 加载参考模型（冻结，用于 KL 散度）
        log("  加载参考模型 (冻结)...")
        ref_model = AutoModelForCausalLM.from_pretrained(
            current_model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True,
        )
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False

        # 加载策略模型 + LoRA
        log("  加载策略模型 + LoRA...")
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

        # 优化器
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, policy_model.parameters()),
            lr=args.lr,
            weight_decay=0.01,
        )

        # ========================================
        # Phase 3: GRPO 训练循环
        # ========================================
        log("\n[Phase 3] GRPO 训练...")
        num_steps = len(all_prompts) // args.batch_size
        log(f"  总步数: {num_steps}")

        epoch_metrics = {"loss": 0, "policy_loss": 0, "kl_divergence": 0, "mean_reward": 0}
        step_count = 0

        for step in range(num_steps):
            batch_start = step * args.batch_size
            batch_end = batch_start + args.batch_size

            batch_prompts = all_prompts[batch_start:batch_end]
            batch_responses = all_responses[batch_start:batch_end]
            batch_rewards = all_rewards[batch_start:batch_end]

            metrics = grpo_train_step(
                model=policy_model,
                ref_model=ref_model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                prompts=batch_prompts,
                grouped_responses=batch_responses,
                grouped_rewards=batch_rewards,
                device=device,
                clip_epsilon=args.clip_epsilon,
                kl_coef=args.kl_coef,
                max_length=args.max_length,
            )

            step_count += 1
            for key in epoch_metrics:
                if key in metrics:
                    epoch_metrics[key] += metrics[key]

            if (step + 1) % 10 == 0 or step == 0:
                log(f"  Step {step+1}/{num_steps} | "
                    f"loss={metrics['loss']:.4f} | "
                    f"policy={metrics['policy_loss']:.4f} | "
                    f"kl={metrics['kl_divergence']:.4f} | "
                    f"reward={metrics['mean_reward']:.3f}")

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
        log(f"\n  Epoch {epoch+1} 平均指标:")
        log(f"    loss={epoch_metrics['loss']:.4f}")
        log(f"    policy_loss={epoch_metrics['policy_loss']:.4f}")
        log(f"    kl_divergence={epoch_metrics['kl_divergence']:.4f}")
        log(f"    mean_reward={epoch_metrics['mean_reward']:.3f}")

        # ========================================
        # Phase 4: 合并 LoRA 并保存
        # ========================================
        log("\n[Phase 4] 合并 LoRA 权重...")
        epoch_output = str(output_dir / f"epoch-{epoch+1}")
        merge_and_save_lora(policy_model, tokenizer, epoch_output)

        # 释放训练模型显存
        del policy_model, ref_model, optimizer
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
