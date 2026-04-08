"""
CUDA 算子 03: RMSNorm (Root Mean Square Normalization)
=====================================================
RMSNorm 是 Qwen2.5、LLaMA 等现代 LLM 使用的归一化层，替代了传统的 LayerNorm。

面试考点:
---------
1. RMSNorm vs LayerNorm 的区别 (去掉了 mean 的计算，更快)
2. 为什么 LLM 用 RMSNorm 而不用 LayerNorm
3. Fused kernel 的优势: 减少 Global Memory 读写次数
4. Reduction 操作在 GPU 上的实现 (求平方和)
5. 数值稳定性: epsilon 的作用

原理:
-----
LayerNorm:  y = (x - mean(x)) / sqrt(var(x) + eps) * weight + bias
RMSNorm:    y = x / sqrt(mean(x^2) + eps) * weight

RMSNorm 的优势:
- 去掉了 mean 和 bias 的计算，减少约 30% 计算量
- 实验表明效果与 LayerNorm 相当
- 更适合 GPU 并行: 只需一次 reduction (求平方和)

在 Qwen2.5-7B 中，每个 Transformer block 有 2 个 RMSNorm:
- Attention 前的 input_layernorm
- FFN 前的 post_attention_layernorm
"""

import torch
import triton
import triton.language as tl


# ============================================
# Triton 实现
# ============================================
@triton.jit
def rmsnorm_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    input_row_stride,
    num_cols,
    epsilon,
    BLOCK_SIZE: tl.constexpr,
):
    """
    RMSNorm kernel: 每个 program instance 处理一行 (一个 token 的 hidden_state)

    计算流程:
    1. 加载一行 x
    2. 计算 mean(x^2) = sum(x^2) / num_cols
    3. 计算 rms = sqrt(mean(x^2) + eps)
    4. 归一化: y = (x / rms) * weight
    """
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < num_cols

    # 加载一行数据
    row_start = input_ptr + row_idx * input_row_stride
    row_data = tl.load(row_start + col_offsets, mask=mask, other=0.0).to(tl.float32)

    # 计算 mean(x^2)
    # 注意: 用 float32 累加避免精度损失 (即使输入是 float16)
    squared = row_data * row_data
    mean_squared = tl.sum(squared, axis=0) / num_cols

    # 计算 1/rms (用 rsqrt 比 sqrt + div 更快)
    inv_rms = 1.0 / tl.sqrt(mean_squared + epsilon)

    # 加载 weight 参数
    weight = tl.load(weight_ptr + col_offsets, mask=mask)

    # 归一化: y = x * inv_rms * weight
    output = row_data * inv_rms * weight

    # 写回
    output_start = output_ptr + row_idx * input_row_stride
    tl.store(output_start + col_offsets, output, mask=mask)


def rmsnorm_triton(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Triton 版本的 RMSNorm"""
    assert input_tensor.is_cuda
    output = torch.empty_like(input_tensor)

    num_rows = input_tensor.shape[0]
    num_cols = input_tensor.shape[-1]

    # 将输入 reshape 为 2D
    input_2d = input_tensor.view(-1, num_cols)
    output_2d = output.view(-1, num_cols)
    total_rows = input_2d.shape[0]

    BLOCK_SIZE = triton.next_power_of_2(num_cols)

    rmsnorm_kernel[(total_rows,)](
        input_2d, weight, output_2d,
        input_2d.stride(0),
        num_cols,
        epsilon,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output


# ============================================
# PyTorch 参考实现
# ============================================
def rmsnorm_pytorch(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """
    PyTorch 原生实现 (用于正确性验证)

    公式: y = x / sqrt(mean(x^2) + eps) * weight
    """
    # 用 float32 计算避免精度损失
    input_float = input_tensor.float()
    # mean(x^2)
    variance = input_float.pow(2).mean(dim=-1, keepdim=True)
    # x / sqrt(mean(x^2) + eps)
    normalized = input_float * torch.rsqrt(variance + epsilon)
    # 乘以 weight
    return (normalized * weight).to(input_tensor.dtype)


# ============================================
# 正确性验证 & 性能对比
# ============================================
def verify_and_benchmark():
    print("=" * 60)
    print("  CUDA 算子 03: RMSNorm")
    print("=" * 60)

    torch.manual_seed(42)

    # 模拟 Qwen2.5-7B 的 hidden_size = 4096
    batch_size, seq_len, hidden_size = 4, 512, 4096
    epsilon = 1e-6

    input_tensor = torch.randn(
        batch_size * seq_len, hidden_size, device="cuda", dtype=torch.float32
    )
    weight = torch.ones(hidden_size, device="cuda", dtype=torch.float32)

    # 正确性验证
    output_pytorch = rmsnorm_pytorch(input_tensor, weight, epsilon)
    output_triton = rmsnorm_triton(input_tensor, weight, epsilon)

    max_diff = (output_pytorch - output_triton).abs().max().item()
    print(f"\n  正确性验证:")
    print(f"    形状: ({batch_size * seq_len}, {hidden_size})")
    print(f"    最大误差: {max_diff:.2e}")
    print(f"    通过: {'✅' if max_diff < 1e-5 else '❌'}")

    # 验证归一化后的 RMS ≈ 1
    rms_after = output_triton.float().pow(2).mean(dim=-1).sqrt()
    weight_rms = weight.float().pow(2).mean().sqrt().item()
    rms_diff = (rms_after.mean().item() - weight_rms)
    print(f"    归一化后 RMS ≈ weight_rms: 偏差 {abs(rms_diff):.4f}")

    # 性能对比
    print(f"\n  性能对比 ({batch_size * seq_len}×{hidden_size}):")

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(1000):
        rmsnorm_pytorch(input_tensor, weight, epsilon)
    end.record()
    torch.cuda.synchronize()
    pytorch_time = start.elapsed_time(end) / 1000

    start.record()
    for _ in range(1000):
        rmsnorm_triton(input_tensor, weight, epsilon)
    end.record()
    torch.cuda.synchronize()
    triton_time = start.elapsed_time(end) / 1000

    print(f"    PyTorch:  {pytorch_time:.4f} ms")
    print(f"    Triton:   {triton_time:.4f} ms")
    print(f"    加速比:   {pytorch_time / triton_time:.2f}x")

    print(f"\n  💡 面试要点:")
    print(f"    - RMSNorm 比 LayerNorm 少一次 reduction (不需要算 mean)")
    print(f"    - Fused kernel 将 x^2 求和、rsqrt、乘 weight 合并为一次 kernel launch")
    print(f"    - 在 LLM 推理中，RMSNorm 是 memory-bound 算子，融合可减少显存带宽压力")


if __name__ == "__main__":
    verify_and_benchmark()
