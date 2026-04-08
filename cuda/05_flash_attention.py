"""
CUDA 算子 05: Flash Attention (简化版)
======================================
FlashAttention 是 LLM 推理和训练中最重要的优化算法，面试必考。

面试考点:
---------
1. 标准 Attention 的显存瓶颈: O(N^2) 的 attention matrix
2. FlashAttention 的核心思想: tiling + online softmax
3. IO 复杂度分析: 从 O(N^2) 降到 O(N^2 * d / M)
4. 为什么 FlashAttention 更快: 减少 HBM 访问，利用 SRAM
5. Online Softmax 在分块计算中的应用
6. FlashAttention-2 的改进: 更好的并行度和 warp 分配

原理:
-----
标准 Attention:
    S = Q @ K^T          # [N, N], 需要 O(N^2) 显存
    P = softmax(S)       # [N, N]
    O = P @ V            # [N, d]

问题: 当 N 很大时 (如 N=8192)，S 和 P 矩阵占用大量 HBM。

FlashAttention 的解决方案:
    1. 将 Q, K, V 分块 (tiling)
    2. 每次只计算一个小块的 attention
    3. 用 Online Softmax 在分块之间正确地累积结果
    4. 不需要存储完整的 N×N attention matrix

分块计算的关键:
    对于第 j 个 K/V 块:
        S_j = Q_block @ K_j^T                    # 局部 attention score
        m_new = max(m_old, rowmax(S_j))           # 更新 running max
        P_j = exp(S_j - m_new)                    # 局部 softmax 分子
        l_new = l_old * exp(m_old - m_new) + rowsum(P_j)  # 更新 running sum
        O = O * (l_old * exp(m_old - m_new) / l_new) + P_j @ V_j / l_new  # 修正累积输出

GPU 内存层次:
    HBM (High Bandwidth Memory): 大但慢 (~1-2 TB/s)
    SRAM (Shared Memory):        小但快 (~19 TB/s)

    FlashAttention 将 Q/K/V 的小块加载到 SRAM 中计算，
    避免将 N×N 的中间结果写回 HBM。
"""

import torch
import triton
import triton.language as tl


# ============================================
# Triton 实现: 简化版 FlashAttention Forward
# ============================================
@triton.jit
def flash_attention_kernel(
    query_ptr, key_ptr, value_ptr, output_ptr,
    softmax_scale,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    seq_len,
    BLOCK_M: tl.constexpr,  # Q 的分块大小
    BLOCK_N: tl.constexpr,  # K/V 的分块大小
    HEAD_DIM: tl.constexpr,
):
    """
    FlashAttention Forward Kernel (简化版，不含 causal mask)

    每个 program instance 处理:
    - 一个 (batch, head) 组合中的一个 Q 块 (BLOCK_M 行)
    - 遍历所有 K/V 块，用 online softmax 累积结果

    这是 FlashAttention 论文 Algorithm 1 的 Triton 实现。
    """
    # 当前处理的 Q 块索引
    block_m_idx = tl.program_id(0)
    # 当前处理的 (batch, head)
    batch_head_idx = tl.program_id(1)

    # 计算 batch 和 head 索引
    # (这里假设 grid 的第二维是 batch * num_heads)
    qkv_offset = batch_head_idx * stride_qh

    # Q 块的行偏移
    m_offsets = block_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    # head_dim 的列偏移
    d_offsets = tl.arange(0, HEAD_DIM)

    # 加载 Q 块: [BLOCK_M, HEAD_DIM]
    q_ptrs = (
        query_ptr
        + qkv_offset
        + m_offsets[:, None] * stride_qm
        + d_offsets[None, :] * stride_qk
    )
    q_mask = m_offsets[:, None] < seq_len
    query_block = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # 初始化 Online Softmax 的状态
    # m_i: running max, 初始化为 -inf
    # l_i: running sum of exp, 初始化为 0
    # acc: 累积输出, 初始化为 0
    running_max = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    running_sum = tl.zeros([BLOCK_M], dtype=tl.float32)
    accumulator = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # 遍历所有 K/V 块 (这是 FlashAttention 的核心循环)
    num_kv_blocks = tl.cdiv(seq_len, BLOCK_N)
    for block_n_idx in range(num_kv_blocks):
        n_offsets = block_n_idx * BLOCK_N + tl.arange(0, BLOCK_N)

        # 加载 K 块: [BLOCK_N, HEAD_DIM]
        k_ptrs = (
            key_ptr
            + qkv_offset
            + n_offsets[:, None] * stride_kn
            + d_offsets[None, :] * stride_kk
        )
        k_mask = n_offsets[:, None] < seq_len
        key_block = tl.load(k_ptrs, mask=k_mask, other=0.0)

        # 加载 V 块: [BLOCK_N, HEAD_DIM]
        v_ptrs = (
            value_ptr
            + qkv_offset
            + n_offsets[:, None] * stride_vn
            + d_offsets[None, :] * stride_vk
        )
        v_mask = n_offsets[:, None] < seq_len
        value_block = tl.load(v_ptrs, mask=v_mask, other=0.0)

        # Step 1: 计算局部 attention score
        # S_ij = Q_i @ K_j^T * scale, shape: [BLOCK_M, BLOCK_N]
        scores = tl.dot(query_block, tl.trans(key_block)) * softmax_scale

        # 对超出 seq_len 的位置 mask 为 -inf
        scores_mask = m_offsets[:, None] < seq_len
        kv_mask = n_offsets[None, :] < seq_len
        scores = tl.where(scores_mask & kv_mask, scores, float("-inf"))

        # Step 2: Online Softmax 更新
        # 2a. 计算当前块的行最大值
        block_max = tl.max(scores, axis=1)  # [BLOCK_M]

        # 2b. 更新 running max
        new_max = tl.maximum(running_max, block_max)

        # 2c. 修正之前的累积值
        #     old_sum 和 old_acc 需要乘以 exp(old_max - new_max) 来修正
        correction = tl.exp(running_max - new_max)
        running_sum = running_sum * correction
        accumulator = accumulator * correction[:, None]

        # 2d. 计算当前块的 exp(scores - new_max)
        exp_scores = tl.exp(scores - new_max[:, None])  # [BLOCK_M, BLOCK_N]

        # 2e. 更新 running sum
        running_sum = running_sum + tl.sum(exp_scores, axis=1)

        # 2f. 累积 P @ V
        accumulator = accumulator + tl.dot(exp_scores.to(value_block.dtype), value_block)

        # 2g. 更新 running max
        running_max = new_max

    # 最终归一化: output = accumulator / running_sum
    output = accumulator / running_sum[:, None]

    # 写回结果
    o_ptrs = (
        output_ptr
        + qkv_offset
        + m_offsets[:, None] * stride_om
        + d_offsets[None, :] * stride_ok
    )
    o_mask = m_offsets[:, None] < seq_len
    tl.store(o_ptrs, output, mask=o_mask)


def flash_attention_triton(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    """
    Triton 版本的 FlashAttention

    Args:
        query: [batch, num_heads, seq_len, head_dim]
        key:   [batch, num_heads, seq_len, head_dim]
        value: [batch, num_heads, seq_len, head_dim]

    Returns:
        output: [batch, num_heads, seq_len, head_dim]
    """
    batch, num_heads, seq_len, head_dim = query.shape
    softmax_scale = head_dim ** -0.5

    output = torch.empty_like(query)

    # 分块大小
    BLOCK_M = 64
    BLOCK_N = 64

    # Grid: (num_q_blocks, batch * num_heads)
    grid = (triton.cdiv(seq_len, BLOCK_M), batch * num_heads)

    flash_attention_kernel[grid](
        query, key, value, output,
        softmax_scale,
        query.stride(0), query.stride(1), query.stride(2), query.stride(3),
        key.stride(0), key.stride(1), key.stride(2), key.stride(3),
        value.stride(0), value.stride(1), value.stride(2), value.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        seq_len,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        HEAD_DIM=head_dim,
    )
    return output


# ============================================
# PyTorch 参考实现 (标准 Attention)
# ============================================
def standard_attention_pytorch(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    """
    标准 Attention 实现 (用于正确性验证)

    S = Q @ K^T / sqrt(d)
    P = softmax(S)
    O = P @ V

    显存: O(N^2) 用于存储 S 和 P
    """
    head_dim = query.shape[-1]
    scale = head_dim ** -0.5

    # [batch, heads, seq, seq] — 这就是 FlashAttention 要避免的 O(N^2) 矩阵
    attention_scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    attention_probs = torch.softmax(attention_scores, dim=-1)
    output = torch.matmul(attention_probs, value)
    return output


# ============================================
# 正确性验证 & 性能对比
# ============================================
def verify_and_benchmark():
    print("=" * 60)
    print("  CUDA 算子 05: FlashAttention (简化版)")
    print("=" * 60)

    torch.manual_seed(42)

    # 模拟 Qwen2.5-7B 的一层 attention
    batch, num_heads, seq_len, head_dim = 1, 4, 512, 64

    query = torch.randn(batch, num_heads, seq_len, head_dim, device="cuda", dtype=torch.float32)
    key = torch.randn(batch, num_heads, seq_len, head_dim, device="cuda", dtype=torch.float32)
    value = torch.randn(batch, num_heads, seq_len, head_dim, device="cuda", dtype=torch.float32)

    # 正确性验证
    output_standard = standard_attention_pytorch(query, key, value)
    output_flash = flash_attention_triton(query, key, value)

    max_diff = (output_standard - output_flash).abs().max().item()
    mean_diff = (output_standard - output_flash).abs().mean().item()

    print(f"\n  正确性验证:")
    print(f"    形状: Q/K/V = ({batch}, {num_heads}, {seq_len}, {head_dim})")
    print(f"    最大误差: {max_diff:.2e}")
    print(f"    平均误差: {mean_diff:.2e}")
    print(f"    通过: {'✅' if max_diff < 1e-2 else '❌'} (FlashAttention 允许较大的数值误差)")

    # 显存对比
    standard_memory = batch * num_heads * seq_len * seq_len * 4  # float32
    flash_memory = batch * num_heads * seq_len * head_dim * 4  # 不需要 N×N 矩阵
    print(f"\n  显存对比 (seq_len={seq_len}):")
    print(f"    标准 Attention: {standard_memory / 1024**2:.1f} MB (存储 N×N attention matrix)")
    print(f"    FlashAttention: {flash_memory / 1024**2:.1f} MB (不存储中间矩阵)")
    print(f"    节省: {(1 - flash_memory / standard_memory) * 100:.0f}%")

    # 长序列下的显存差异更明显
    print(f"\n  长序列显存对比:")
    for length in [1024, 4096, 8192, 16384]:
        std_mem = batch * num_heads * length * length * 4 / 1024**2
        flash_mem = batch * num_heads * length * head_dim * 4 / 1024**2
        print(f"    seq_len={length:>5}: 标准={std_mem:>8.1f} MB, Flash={flash_mem:>6.1f} MB")

    # 性能对比
    print(f"\n  性能对比:")
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(200):
        standard_attention_pytorch(query, key, value)
    end.record()
    torch.cuda.synchronize()
    standard_time = start.elapsed_time(end) / 200

    start.record()
    for _ in range(200):
        flash_attention_triton(query, key, value)
    end.record()
    torch.cuda.synchronize()
    flash_time = start.elapsed_time(end) / 200

    print(f"    标准 Attention: {standard_time:.4f} ms")
    print(f"    FlashAttention: {flash_time:.4f} ms")
    print(f"    加速比: {standard_time / flash_time:.2f}x")

    print(f"\n  💡 面试要点:")
    print(f"    1. FlashAttention 的核心是 tiling + online softmax")
    print(f"    2. 不存储 N×N attention matrix，显存从 O(N^2) 降到 O(N)")
    print(f"    3. IO 复杂度从 O(N^2 * d) 降到 O(N^2 * d^2 / M)")
    print(f"       其中 M 是 SRAM 大小")
    print(f"    4. FlashAttention-2 改进: 减少非 matmul 的 FLOPs，")
    print(f"       更好的 warp 间并行 (沿 seq_len 维度并行)")
    print(f"    5. FlashAttention-3 (Hopper): 利用 TMA 和 wgmma 指令")


if __name__ == "__main__":
    verify_and_benchmark()
