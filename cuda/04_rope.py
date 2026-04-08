"""
CUDA 算子 04: RoPE (Rotary Position Embedding, 旋转位置编码)
============================================================
RoPE 是所有现代 LLM (Qwen, LLaMA, GPT-NeoX 等) 使用的位置编码方式。

面试考点:
---------
1. RoPE 的数学原理: 将位置信息编码为旋转矩阵
2. 为什么 RoPE 优于绝对位置编码和相对位置编码
3. RoPE 的关键性质: 内积只依赖相对位置
4. 实现中的 complex number trick (将实数对视为复数)
5. NTK-aware RoPE / YaRN 等长度外推方法

原理:
-----
RoPE 将 query/key 向量的每两个相邻维度视为一个 2D 平面上的向量，
然后根据 token 位置旋转这个向量:

    对于位置 pos，维度对 (2i, 2i+1):
    theta_i = 10000^(-2i/d)

    [q_{2i}  ]     [cos(pos * theta_i)  -sin(pos * theta_i)] [q_{2i}  ]
    [q_{2i+1}]  =  [sin(pos * theta_i)   cos(pos * theta_i)] [q_{2i+1}]

等价于复数乘法:
    (q_{2i} + j * q_{2i+1}) * (cos(pos * theta_i) + j * sin(pos * theta_i))
    = (q_{2i} + j * q_{2i+1}) * e^(j * pos * theta_i)

关键性质:
    <RoPE(q, m), RoPE(k, n)> = <q, k> 的函数只依赖 (m - n)
    即: 旋转后的内积只取决于相对位置，天然具备相对位置编码的能力。
"""

import torch
import triton
import triton.language as tl


# ============================================
# Triton 实现
# ============================================
@triton.jit
def rope_kernel(
    query_ptr,
    key_ptr,
    cos_ptr,
    sin_ptr,
    output_query_ptr,
    output_key_ptr,
    seq_len,
    num_heads,
    head_dim,
    stride_batch,
    stride_seq,
    stride_head,
    BLOCK_SIZE: tl.constexpr,
):
    """
    RoPE kernel: 对 query 和 key 应用旋转位置编码

    每个 program instance 处理一个 (batch, seq_pos, head) 的 head_dim 维向量。

    旋转公式 (对每对相邻维度):
        out[2i]   = x[2i] * cos - x[2i+1] * sin
        out[2i+1] = x[2i] * sin + x[2i+1] * cos
    """
    # 计算当前处理的 (batch, seq, head) 索引
    pid = tl.program_id(0)
    batch_idx = pid // (seq_len * num_heads)
    remaining = pid % (seq_len * num_heads)
    seq_idx = remaining // num_heads
    head_idx = remaining % num_heads

    # 半维度: RoPE 每次处理一对 (2i, 2i+1)
    half_dim = head_dim // 2
    dim_offsets = tl.arange(0, BLOCK_SIZE)
    mask = dim_offsets < half_dim

    # 计算基地址偏移
    base_offset = (
        batch_idx * stride_batch
        + seq_idx * stride_seq
        + head_idx * stride_head
    )

    # 加载 cos, sin (shape: [seq_len, half_dim])
    cos_offset = seq_idx * half_dim + dim_offsets
    cos_val = tl.load(cos_ptr + cos_offset, mask=mask)
    sin_val = tl.load(sin_ptr + cos_offset, mask=mask)

    # --- 处理 Query ---
    # 加载 q 的前半部分 [0, half_dim) 和后半部分 [half_dim, head_dim)
    query_first_half = tl.load(query_ptr + base_offset + dim_offsets, mask=mask)
    query_second_half = tl.load(
        query_ptr + base_offset + half_dim + dim_offsets, mask=mask
    )

    # 旋转: (a + jb) * (cos + j*sin) = (a*cos - b*sin) + j*(a*sin + b*cos)
    out_query_first = query_first_half * cos_val - query_second_half * sin_val
    out_query_second = query_first_half * sin_val + query_second_half * cos_val

    tl.store(output_query_ptr + base_offset + dim_offsets, out_query_first, mask=mask)
    tl.store(
        output_query_ptr + base_offset + half_dim + dim_offsets,
        out_query_second,
        mask=mask,
    )

    # --- 处理 Key (同样的旋转) ---
    key_first_half = tl.load(key_ptr + base_offset + dim_offsets, mask=mask)
    key_second_half = tl.load(
        key_ptr + base_offset + half_dim + dim_offsets, mask=mask
    )

    out_key_first = key_first_half * cos_val - key_second_half * sin_val
    out_key_second = key_first_half * sin_val + key_second_half * cos_val

    tl.store(output_key_ptr + base_offset + dim_offsets, out_key_first, mask=mask)
    tl.store(
        output_key_ptr + base_offset + half_dim + dim_offsets,
        out_key_second,
        mask=mask,
    )


def precompute_freqs(head_dim: int, max_seq_len: int, theta: float = 10000.0):
    """
    预计算 RoPE 的 cos/sin 频率表

    theta_i = 1 / (10000^(2i/d)), i = 0, 1, ..., d/2 - 1
    freqs[pos, i] = pos * theta_i
    """
    # 频率: theta_i = 1 / (base^(2i/d))
    dim_indices = torch.arange(0, head_dim // 2, dtype=torch.float32)
    freqs = 1.0 / (theta ** (2 * dim_indices / head_dim))

    # 位置 × 频率
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    angles = torch.outer(positions, freqs)  # [seq_len, head_dim//2]

    cos_table = angles.cos()
    sin_table = angles.sin()
    return cos_table, sin_table


def rope_triton(
    query: torch.Tensor,
    key: torch.Tensor,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Triton 版本的 RoPE

    Args:
        query: [batch, seq_len, num_heads, head_dim]
        key:   [batch, seq_len, num_heads, head_dim]
        cos_table: [seq_len, head_dim//2]
        sin_table: [seq_len, head_dim//2]
    """
    batch, seq_len, num_heads, head_dim = query.shape
    assert head_dim % 2 == 0

    output_query = torch.empty_like(query)
    output_key = torch.empty_like(key)

    BLOCK_SIZE = triton.next_power_of_2(head_dim // 2)
    total_programs = batch * seq_len * num_heads

    rope_kernel[(total_programs,)](
        query, key,
        cos_table.to(query.device),
        sin_table.to(query.device),
        output_query, output_key,
        seq_len, num_heads, head_dim,
        query.stride(0), query.stride(1), query.stride(2),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output_query, output_key


# ============================================
# PyTorch 参考实现
# ============================================
def rope_pytorch(
    query: torch.Tensor,
    key: torch.Tensor,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    PyTorch 原生实现 (用于正确性验证)

    使用 "split half" 方式:
    将 head_dim 分为前半和后半，分别应用旋转。
    """
    half_dim = query.shape[-1] // 2
    seq_len = query.shape[1]

    cos = cos_table[:seq_len].unsqueeze(0).unsqueeze(2)  # [1, seq, 1, half_dim]
    sin = sin_table[:seq_len].unsqueeze(0).unsqueeze(2)

    cos = cos.to(query.device, dtype=query.dtype)
    sin = sin.to(query.device, dtype=query.dtype)

    def apply_rotary(x):
        x_first = x[..., :half_dim]
        x_second = x[..., half_dim:]
        rotated_first = x_first * cos - x_second * sin
        rotated_second = x_first * sin + x_second * cos
        return torch.cat([rotated_first, rotated_second], dim=-1)

    return apply_rotary(query), apply_rotary(key)


# ============================================
# 正确性验证 & 性能对比
# ============================================
def verify_and_benchmark():
    print("=" * 60)
    print("  CUDA 算子 04: RoPE (旋转位置编码)")
    print("=" * 60)

    torch.manual_seed(42)

    # 模拟 Qwen2.5-7B: num_heads=32, head_dim=128
    batch, seq_len, num_heads, head_dim = 2, 256, 32, 128

    query = torch.randn(batch, seq_len, num_heads, head_dim, device="cuda", dtype=torch.float32)
    key = torch.randn(batch, seq_len, num_heads, head_dim, device="cuda", dtype=torch.float32)

    # 预计算频率表
    cos_table, sin_table = precompute_freqs(head_dim, seq_len)

    # 正确性验证
    out_q_pytorch, out_k_pytorch = rope_pytorch(query, key, cos_table, sin_table)
    out_q_triton, out_k_triton = rope_triton(query, key, cos_table, sin_table)

    q_diff = (out_q_pytorch - out_q_triton).abs().max().item()
    k_diff = (out_k_pytorch - out_k_triton).abs().max().item()

    print(f"\n  正确性验证:")
    print(f"    形状: ({batch}, {seq_len}, {num_heads}, {head_dim})")
    print(f"    Query 最大误差: {q_diff:.2e} {'✅' if q_diff < 1e-4 else '❌'}")
    print(f"    Key 最大误差:   {k_diff:.2e} {'✅' if k_diff < 1e-4 else '❌'}")

    # 验证 RoPE 的关键性质: 旋转后内积只依赖相对位置
    print(f"\n  RoPE 性质验证 (内积只依赖相对位置):")
    pos_a, pos_b = 10, 15  # 相对距离 = 5
    pos_c, pos_d = 20, 25  # 相对距离 = 5
    dot_ab = (out_q_pytorch[0, pos_a, 0] * out_k_pytorch[0, pos_b, 0]).sum().item()
    dot_cd = (out_q_pytorch[0, pos_c, 0] * out_k_pytorch[0, pos_d, 0]).sum().item()
    print(f"    <q[10], k[15]> = {dot_ab:.4f}")
    print(f"    <q[20], k[25]> = {dot_cd:.4f}")
    print(f"    (相同相对距离=5，内积应接近)")

    # 性能对比
    print(f"\n  性能对比:")
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(500):
        rope_pytorch(query, key, cos_table, sin_table)
    end.record()
    torch.cuda.synchronize()
    pytorch_time = start.elapsed_time(end) / 500

    start.record()
    for _ in range(500):
        rope_triton(query, key, cos_table, sin_table)
    end.record()
    torch.cuda.synchronize()
    triton_time = start.elapsed_time(end) / 500

    print(f"    PyTorch:  {pytorch_time:.4f} ms")
    print(f"    Triton:   {triton_time:.4f} ms")
    print(f"    加速比:   {pytorch_time / triton_time:.2f}x")

    print(f"\n  💡 面试要点:")
    print(f"    - RoPE 将位置编码为旋转角度，内积天然具备相对位置信息")
    print(f"    - 实现上可以用复数乘法或 cos/sin 展开两种方式")
    print(f"    - NTK-aware RoPE 通过修改 base 频率实现长度外推")
    print(f"    - YaRN 在 NTK 基础上加入注意力缩放因子")


if __name__ == "__main__":
    verify_and_benchmark()
