"""
CUDA 算子 06: Fused Residual Add + RMSNorm (算子融合)
====================================================
算子融合 (Kernel Fusion) 是 LLM 推理优化的核心技术之一。
本文件展示如何将 Residual Add 和 RMSNorm 融合为一个 kernel。

面试考点:
---------
1. 为什么要做算子融合: 减少 Global Memory 读写 (memory-bound 优化)
2. Roofline Model: 区分 compute-bound 和 memory-bound 算子
3. LLM 中哪些算子适合融合
4. 融合前后的 IO 量对比
5. vLLM 中的 fused kernels 有哪些

原理:
-----
在 Transformer 中，每个子层 (Attention / FFN) 后都有:
    hidden_states = residual + sublayer_output    # Residual Add
    hidden_states = RMSNorm(hidden_states)        # RMSNorm

不融合 (2 个 kernel):
    Kernel 1 (Add):     读 residual + sublayer_output, 写 hidden_states  → 3N 次 IO
    Kernel 2 (RMSNorm): 读 hidden_states, 写 output                     → 2N 次 IO
    总 IO: 5N 次 (N = 元素总数)

融合 (1 个 kernel):
    Kernel (Fused):     读 residual + sublayer_output, 写 output          → 3N 次 IO
    总 IO: 3N 次

    节省 40% 的 Global Memory 读写!

对于 memory-bound 算子 (如 RMSNorm)，减少 IO 直接等于加速。
"""

import torch
import triton
import triton.language as tl


# ============================================
# Triton 实现: Fused Residual Add + RMSNorm
# ============================================
@triton.jit
def fused_add_rmsnorm_kernel(
    residual_ptr,
    sublayer_output_ptr,
    weight_ptr,
    output_ptr,
    # 同时输出 add 的结果，供下一层使用
    residual_output_ptr,
    row_stride,
    num_cols,
    epsilon,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel: output = RMSNorm(residual + sublayer_output)

    同时输出:
    - residual_output = residual + sublayer_output (供下一个残差连接使用)
    - output = RMSNorm(residual_output) * weight

    一个 kernel 完成两个操作，减少 40% 的显存带宽消耗。
    """
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < num_cols

    # Step 1: 加载 residual 和 sublayer_output
    base_offset = row_idx * row_stride
    residual = tl.load(residual_ptr + base_offset + col_offsets, mask=mask, other=0.0)
    sublayer = tl.load(sublayer_output_ptr + base_offset + col_offsets, mask=mask, other=0.0)

    # Step 2: Residual Add (在寄存器中完成，不写回 Global Memory)
    hidden = (residual + sublayer).to(tl.float32)

    # Step 3: 保存 residual add 的结果 (供下一层残差连接使用)
    tl.store(residual_output_ptr + base_offset + col_offsets, hidden, mask=mask)

    # Step 4: RMSNorm (直接在寄存器中的 hidden 上计算)
    squared = hidden * hidden
    mean_squared = tl.sum(squared, axis=0) / num_cols
    inv_rms = 1.0 / tl.sqrt(mean_squared + epsilon)

    # Step 5: 加载 weight 并归一化
    weight = tl.load(weight_ptr + col_offsets, mask=mask)
    output = hidden * inv_rms * weight

    # Step 6: 写回最终结果
    tl.store(output_ptr + base_offset + col_offsets, output, mask=mask)


def fused_add_rmsnorm_triton(
    residual: torch.Tensor,
    sublayer_output: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Triton 版本的 Fused Residual Add + RMSNorm

    Returns:
        output: RMSNorm(residual + sublayer_output) * weight
        new_residual: residual + sublayer_output (供下一层使用)
    """
    assert residual.shape == sublayer_output.shape
    num_cols = residual.shape[-1]

    input_2d = residual.view(-1, num_cols)
    sublayer_2d = sublayer_output.view(-1, num_cols)
    total_rows = input_2d.shape[0]

    output = torch.empty_like(input_2d)
    new_residual = torch.empty_like(input_2d)

    BLOCK_SIZE = triton.next_power_of_2(num_cols)

    fused_add_rmsnorm_kernel[(total_rows,)](
        input_2d, sublayer_2d, weight,
        output, new_residual,
        input_2d.stride(0),
        num_cols,
        epsilon,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output.view_as(residual), new_residual.view_as(residual)


# ============================================
# PyTorch 参考实现 (非融合版本)
# ============================================
def unfused_add_rmsnorm_pytorch(
    residual: torch.Tensor,
    sublayer_output: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    非融合版本: 分两步执行

    这是"正常"的 PyTorch 写法，会产生中间张量。
    """
    # Step 1: Residual Add (产生中间张量，写入 Global Memory)
    hidden = residual + sublayer_output

    # Step 2: RMSNorm (从 Global Memory 读取中间张量)
    hidden_float = hidden.float()
    variance = hidden_float.pow(2).mean(dim=-1, keepdim=True)
    normalized = hidden_float * torch.rsqrt(variance + epsilon)
    output = (normalized * weight).to(residual.dtype)

    return output, hidden


# ============================================
# 正确性验证 & 性能对比
# ============================================
def verify_and_benchmark():
    print("=" * 60)
    print("  CUDA 算子 06: Fused Residual Add + RMSNorm")
    print("=" * 60)

    torch.manual_seed(42)

    # 模拟 Qwen2.5-7B
    batch_seq = 2048  # batch * seq_len
    hidden_size = 4096
    epsilon = 1e-6

    residual = torch.randn(batch_seq, hidden_size, device="cuda", dtype=torch.float32)
    sublayer_output = torch.randn(batch_seq, hidden_size, device="cuda", dtype=torch.float32)
    weight = torch.ones(hidden_size, device="cuda", dtype=torch.float32)

    # 正确性验证
    out_unfused, res_unfused = unfused_add_rmsnorm_pytorch(
        residual, sublayer_output, weight, epsilon
    )
    out_fused, res_fused = fused_add_rmsnorm_triton(
        residual, sublayer_output, weight, epsilon
    )

    out_diff = (out_unfused - out_fused).abs().max().item()
    res_diff = (res_unfused - res_fused).abs().max().item()

    print(f"\n  正确性验证:")
    print(f"    形状: ({batch_seq}, {hidden_size})")
    print(f"    Output 最大误差:   {out_diff:.2e} {'✅' if out_diff < 1e-4 else '❌'}")
    print(f"    Residual 最大误差: {res_diff:.2e} {'✅' if res_diff < 1e-4 else '❌'}")

    # IO 量分析
    element_size = 4  # float32 = 4 bytes
    total_elements = batch_seq * hidden_size
    unfused_io = 5 * total_elements * element_size  # 读3写2 (add: 读2写1, rmsnorm: 读1写1)
    fused_io = 4 * total_elements * element_size     # 读2写2 (读 residual+sublayer, 写 output+new_residual)

    print(f"\n  IO 量分析:")
    print(f"    非融合 (2 kernels): {unfused_io / 1024**2:.1f} MB")
    print(f"    融合 (1 kernel):    {fused_io / 1024**2:.1f} MB")
    print(f"    IO 节省:            {(1 - fused_io / unfused_io) * 100:.0f}%")

    # 性能对比
    print(f"\n  性能对比 ({batch_seq}×{hidden_size}):")

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # 非融合版本
    start.record()
    for _ in range(1000):
        unfused_add_rmsnorm_pytorch(residual, sublayer_output, weight, epsilon)
    end.record()
    torch.cuda.synchronize()
    unfused_time = start.elapsed_time(end) / 1000

    # 融合版本
    start.record()
    for _ in range(1000):
        fused_add_rmsnorm_triton(residual, sublayer_output, weight, epsilon)
    end.record()
    torch.cuda.synchronize()
    fused_time = start.elapsed_time(end) / 1000

    print(f"    非融合 (PyTorch): {unfused_time:.4f} ms")
    print(f"    融合 (Triton):    {fused_time:.4f} ms")
    print(f"    加速比:           {unfused_time / fused_time:.2f}x")

    print(f"\n  💡 面试要点:")
    print(f"    1. 算子融合的本质: 减少 Global Memory (HBM) 的读写次数")
    print(f"    2. 对 memory-bound 算子效果最好 (如 RMSNorm, Add, Activation)")
    print(f"    3. 对 compute-bound 算子效果有限 (如 GEMM)")
    print(f"    4. vLLM 中的融合算子: fused_add_rmsnorm, fused_moe, silu_and_mul")
    print(f"    5. Roofline Model: 算术强度 < 硬件拐点 → memory-bound → 适合融合")


if __name__ == "__main__":
    verify_and_benchmark()
