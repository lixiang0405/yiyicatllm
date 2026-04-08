"""
CUDA 算子 02: Softmax (含 Online Softmax)
==========================================
Softmax 是 Transformer Attention 的核心组件，也是 FlashAttention 的基础。

面试考点:
---------
1. Naive Softmax 的数值稳定性问题 (减最大值)
2. Online Softmax 算法 (单次遍历，FlashAttention 的核心)
3. Safe Softmax: softmax(x) = softmax(x - max(x))
4. Shared Memory 的使用: block 内的 reduction 操作
5. Warp-level primitives: __shfl_down_sync 等

原理:
-----
标准 Softmax:
    softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))

需要两次遍历:
    Pass 1: 找 max(x)
    Pass 2: 计算 exp 和 sum，然后归一化

Online Softmax (Milakov & Gimelshein, 2018):
    单次遍历同时维护 running max 和 running sum:
    当遇到新的 max 时，修正之前累积的 sum:
        new_sum = old_sum * exp(old_max - new_max) + exp(x_i - new_max)
    这是 FlashAttention 能分块计算的数学基础。
"""

import torch
import triton
import triton.language as tl


# ============================================
# Triton 实现: Safe Softmax (行级)
# ============================================
@triton.jit
def softmax_kernel(
    input_ptr,
    output_ptr,
    num_cols,
    input_row_stride,
    output_row_stride,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Softmax kernel: 每个 program instance 处理一行

    算法步骤:
    1. 加载一行数据
    2. 找到行最大值 (数值稳定性)
    3. 计算 exp(x - max)
    4. 求和
    5. 归一化: exp(x - max) / sum
    """
    # 当前处理第几行
    row_idx = tl.program_id(0)

    # 计算当前行的起始地址
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < num_cols

    # Step 1: 加载一行数据
    row_data = tl.load(row_start_ptr + col_offsets, mask=mask, other=float("-inf"))

    # Step 2: 找行最大值 (Safe Softmax 的关键)
    row_max = tl.max(row_data, axis=0)

    # Step 3: 计算 exp(x - max)，减去 max 防止 exp 溢出
    numerator = tl.exp(row_data - row_max)

    # Step 4: 求和
    denominator = tl.sum(numerator, axis=0)

    # Step 5: 归一化
    softmax_output = numerator / denominator

    # 写回
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    tl.store(output_row_start_ptr + col_offsets, softmax_output, mask=mask)


def softmax_triton(input_tensor: torch.Tensor) -> torch.Tensor:
    """Triton 版本的 Softmax (按最后一维)"""
    assert input_tensor.is_cuda
    num_rows, num_cols = input_tensor.shape

    # BLOCK_SIZE 必须是 2 的幂且 >= num_cols
    BLOCK_SIZE = triton.next_power_of_2(num_cols)

    output = torch.empty_like(input_tensor)

    # 每行一个 program instance
    grid = (num_rows,)

    softmax_kernel[grid](
        input_tensor, output,
        num_cols,
        input_tensor.stride(0),
        output.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output


# ============================================
# Online Softmax (纯 Python 演示，理解原理用)
# ============================================
def online_softmax_demo(row: torch.Tensor) -> torch.Tensor:
    """
    Online Softmax: 单次遍历计算 softmax

    这是 FlashAttention 的数学基础。
    在标准实现中需要两次遍历 (找 max + 计算 exp/sum)，
    Online Softmax 只需一次遍历。

    面试时能手写这个算法是很大的加分项。
    """
    running_max = float("-inf")
    running_sum = 0.0

    # 单次遍历: 同时维护 max 和 sum
    for element in row:
        element_val = element.item()
        if element_val > running_max:
            # 遇到新的 max，修正之前的 sum
            # 关键公式: sum = sum * exp(old_max - new_max) + exp(x - new_max)
            running_sum = running_sum * torch.exp(
                torch.tensor(running_max - element_val)
            ).item()
            running_max = element_val
        running_sum += torch.exp(torch.tensor(element_val - running_max)).item()

    # 最终归一化
    result = torch.exp(row - running_max) / running_sum
    return result


# ============================================
# PyTorch 参考实现
# ============================================
def softmax_pytorch(input_tensor: torch.Tensor) -> torch.Tensor:
    """PyTorch 原生实现"""
    return torch.softmax(input_tensor, dim=-1)


# ============================================
# 正确性验证 & 性能对比
# ============================================
def verify_and_benchmark():
    print("=" * 60)
    print("  CUDA 算子 02: Softmax (含 Online Softmax)")
    print("=" * 60)

    torch.manual_seed(42)

    # --- Online Softmax 正确性验证 ---
    print("\n  [Online Softmax 验证]")
    small_row = torch.randn(8)
    online_result = online_softmax_demo(small_row)
    pytorch_result = torch.softmax(small_row, dim=-1)
    max_diff = (online_result - pytorch_result).abs().max().item()
    print(f"    输入: {small_row.tolist()[:4]}...")
    print(f"    Online:  {online_result.tolist()[:4]}...")
    print(f"    PyTorch: {pytorch_result.tolist()[:4]}...")
    print(f"    最大误差: {max_diff:.2e} {'✅' if max_diff < 1e-5 else '❌'}")

    # --- Triton Softmax 验证 ---
    print("\n  [Triton Softmax 验证]")
    num_rows, num_cols = 1024, 2048
    input_tensor = torch.randn(num_rows, num_cols, device="cuda", dtype=torch.float32)

    output_pytorch = softmax_pytorch(input_tensor)
    output_triton = softmax_triton(input_tensor)

    max_diff = (output_pytorch - output_triton).abs().max().item()
    print(f"    形状: ({num_rows}, {num_cols})")
    print(f"    最大误差: {max_diff:.2e}")
    print(f"    通过: {'✅' if max_diff < 1e-5 else '❌'}")

    # 验证概率和为 1
    row_sums = output_triton.sum(dim=-1)
    sum_diff = (row_sums - 1.0).abs().max().item()
    print(f"    行和=1 验证: 最大偏差 {sum_diff:.2e} {'✅' if sum_diff < 1e-5 else '❌'}")

    # --- 性能对比 ---
    print(f"\n  性能对比 ({num_rows}×{num_cols}):")

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(1000):
        softmax_pytorch(input_tensor)
    end.record()
    torch.cuda.synchronize()
    pytorch_time = start.elapsed_time(end) / 1000

    start.record()
    for _ in range(1000):
        softmax_triton(input_tensor)
    end.record()
    torch.cuda.synchronize()
    triton_time = start.elapsed_time(end) / 1000

    print(f"    PyTorch:  {pytorch_time:.4f} ms")
    print(f"    Triton:   {triton_time:.4f} ms")
    print(f"    加速比:   {pytorch_time / triton_time:.2f}x")


if __name__ == "__main__":
    verify_and_benchmark()
