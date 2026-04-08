"""
CUDA 算子 01: Vector Add (向量加法)
==================================
这是 CUDA 编程的 "Hello World"，用于理解 GPU 并行计算的基本概念。

面试考点:
---------
1. GPU 线程层次: Grid → Block → Thread
2. 线程索引计算: global_id = blockIdx.x * blockDim.x + threadIdx.x
3. 边界检查: 防止越界访问
4. 内存模型: Global Memory 的合并访问 (Coalesced Access)
5. BLOCK_SIZE 的选择: 通常为 32 的倍数 (一个 warp = 32 threads)

原理:
-----
每个 GPU 线程负责计算输出向量中的一个元素:
    output[i] = input_a[i] + input_b[i]

N 个元素被分配到 ceil(N / BLOCK_SIZE) 个 block 中，
每个 block 包含 BLOCK_SIZE 个线程并行执行。
"""

import torch
import triton
import triton.language as tl


# ============================================
# Triton 实现
# ============================================
@triton.jit
def vector_add_kernel(
    # 指针参数: 指向输入/输出张量的首地址
    input_a_ptr,
    input_b_ptr,
    output_ptr,
    # 元数据参数
    num_elements,
    # 编译时常量: block 大小
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel: 向量加法

    每个 program instance (类似 CUDA 的一个 block) 处理 BLOCK_SIZE 个元素。
    """
    # 1. 计算当前 program 处理的元素范围
    #    pid 类似 CUDA 的 blockIdx.x
    program_id = tl.program_id(axis=0)

    # 2. 计算当前 block 负责的元素偏移量
    #    类似 CUDA 的 threadIdx.x + blockIdx.x * blockDim.x
    block_start = program_id * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # 3. 边界检查: 最后一个 block 可能不满
    mask = offsets < num_elements

    # 4. 从 Global Memory 加载数据 (合并访问)
    input_a = tl.load(input_a_ptr + offsets, mask=mask)
    input_b = tl.load(input_b_ptr + offsets, mask=mask)

    # 5. 计算
    output = input_a + input_b

    # 6. 写回 Global Memory
    tl.store(output_ptr + offsets, output, mask=mask)


def vector_add_triton(input_a: torch.Tensor, input_b: torch.Tensor) -> torch.Tensor:
    """Triton 版本的向量加法"""
    assert input_a.shape == input_b.shape
    assert input_a.is_cuda

    output = torch.empty_like(input_a)
    num_elements = input_a.numel()

    # Grid 大小: 需要多少个 program instance (block)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(num_elements, BLOCK_SIZE),)

    # 启动 kernel
    vector_add_kernel[grid](
        input_a, input_b, output,
        num_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output


# ============================================
# PyTorch 参考实现
# ============================================
def vector_add_pytorch(input_a: torch.Tensor, input_b: torch.Tensor) -> torch.Tensor:
    """PyTorch 原生实现 (用于正确性验证)"""
    return input_a + input_b


# ============================================
# 正确性验证 & 性能对比
# ============================================
def verify_and_benchmark():
    print("=" * 60)
    print("  CUDA 算子 01: Vector Add")
    print("=" * 60)

    torch.manual_seed(42)
    num_elements = 1024 * 1024  # 1M 元素

    input_a = torch.randn(num_elements, device="cuda", dtype=torch.float32)
    input_b = torch.randn(num_elements, device="cuda", dtype=torch.float32)

    # 正确性验证
    output_pytorch = vector_add_pytorch(input_a, input_b)
    output_triton = vector_add_triton(input_a, input_b)

    max_diff = (output_pytorch - output_triton).abs().max().item()
    print(f"\n  正确性验证:")
    print(f"    最大误差: {max_diff:.2e}")
    print(f"    通过: {'✅' if max_diff < 1e-6 else '❌'}")

    # 性能对比
    print(f"\n  性能对比 (N={num_elements:,}):")

    # PyTorch
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(1000):
        vector_add_pytorch(input_a, input_b)
    end.record()
    torch.cuda.synchronize()
    pytorch_time = start.elapsed_time(end) / 1000
    print(f"    PyTorch:  {pytorch_time:.4f} ms")

    # Triton
    start.record()
    for _ in range(1000):
        vector_add_triton(input_a, input_b)
    end.record()
    torch.cuda.synchronize()
    triton_time = start.elapsed_time(end) / 1000
    print(f"    Triton:   {triton_time:.4f} ms")
    print(f"    加速比:   {pytorch_time / triton_time:.2f}x")


if __name__ == "__main__":
    verify_and_benchmark()
