/*
 * CUDA 算子: Softmax (Safe Softmax + Online Softmax)
 * ===================================================
 * Softmax 是 Attention 的核心组件，理解 Online Softmax 是理解 FlashAttention 的前提。
 *
 * 面试考点:
 * ---------
 * 1. 数值稳定性: 为什么要减最大值 (防止 exp 溢出)
 * 2. Warp-level Reduction: __shfl_down_sync 求 max 和 sum
 * 3. Online Softmax: 单次遍历，FlashAttention 的数学基础
 * 4. Block-level Reduction: Shared Memory + Warp Reduction
 * 5. 向量化访存: float4 提升带宽利用率
 *
 * 编译运行:
 *   nvcc -O3 -arch=sm_89 softmax.cu -o softmax && ./softmax
 */

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                 \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

#define WARP_SIZE 32

// ============================================
// Warp-level Reduction 原语
// ============================================
/*
 * __shfl_down_sync: warp 内线程间直接交换寄存器数据
 * 比 Shared Memory 更快 (寄存器级别，零延迟)
 *
 * 工作原理 (以 max 为例，warp_size=32):
 *   Step 1: offset=16, 线程 0 和 16 比较, 1 和 17 比较, ...
 *   Step 2: offset=8,  线程 0 和 8 比较, 1 和 9 比较, ...
 *   Step 3: offset=4,  ...
 *   Step 4: offset=2,  ...
 *   Step 5: offset=1,  线程 0 和 1 比较
 *   最终: 线程 0 持有整个 warp 的最大值
 */
__device__ float warp_reduce_max(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// ============================================
// Version 1: Naive Safe Softmax (每行一个 block)
// ============================================
/*
 * 三次遍历:
 *   Pass 1: 找 max (数值稳定性)
 *   Pass 2: 计算 exp(x - max) 和 sum
 *   Pass 3: 归一化 exp(x - max) / sum
 *
 * 使用 Shared Memory 做 block-level reduction
 */
__global__ void softmax_naive(
    const float* __restrict__ input,   // [num_rows, num_cols]
    float* __restrict__ output,
    int num_rows, int num_cols
) {
    extern __shared__ float shared_data[];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    if (row >= num_rows) return;

    const float* row_input = input + row * num_cols;
    float* row_output = output + row * num_cols;

    // Pass 1: 找行最大值 (每个线程处理多个元素)
    float thread_max = -FLT_MAX;
    for (int col = tid; col < num_cols; col += block_size) {
        thread_max = fmaxf(thread_max, row_input[col]);
    }

    // Warp-level reduction
    thread_max = warp_reduce_max(thread_max);

    // Block-level reduction via shared memory
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int num_warps = (block_size + WARP_SIZE - 1) / WARP_SIZE;

    if (lane_id == 0) {
        shared_data[warp_id] = thread_max;
    }
    __syncthreads();

    // 第一个 warp 做最终 reduction
    if (warp_id == 0) {
        float val = (lane_id < num_warps) ? shared_data[lane_id] : -FLT_MAX;
        val = warp_reduce_max(val);
        if (lane_id == 0) {
            shared_data[0] = val;  // 存储全局 max
        }
    }
    __syncthreads();
    float row_max = shared_data[0];

    // Pass 2: 计算 exp(x - max) 和 sum
    float thread_sum = 0.0f;
    for (int col = tid; col < num_cols; col += block_size) {
        float exp_val = expf(row_input[col] - row_max);
        row_output[col] = exp_val;  // 暂存 exp 值
        thread_sum += exp_val;
    }

    // Reduce sum
    thread_sum = warp_reduce_sum(thread_sum);
    if (lane_id == 0) {
        shared_data[warp_id] = thread_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < num_warps) ? shared_data[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane_id == 0) {
            shared_data[0] = val;
        }
    }
    __syncthreads();
    float row_sum = shared_data[0];

    // Pass 3: 归一化
    for (int col = tid; col < num_cols; col += block_size) {
        row_output[col] = row_output[col] / row_sum;
    }
}

// ============================================
// Version 2: Online Softmax (两次遍历优化为一次)
// ============================================
/*
 * Online Softmax 算法 (Milakov & Gimelshein, 2018):
 *
 * 核心思想: 在遍历数据时同时维护 running_max 和 running_sum
 * 当遇到新的 max 时，修正之前的 sum:
 *   new_sum = old_sum * exp(old_max - new_max) + exp(x_i - new_max)
 *
 * 这是 FlashAttention 能分块计算 attention 的数学基础!
 *
 * 优化: 从 3 次遍历减少到 2 次 (1次 online pass + 1次归一化)
 */
__global__ void softmax_online(
    const float* __restrict__ input,
    float* __restrict__ output,
    int num_rows, int num_cols
) {
    extern __shared__ float shared_data[];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    if (row >= num_rows) return;

    const float* row_input = input + row * num_cols;
    float* row_output = output + row * num_cols;

    // Online Pass: 单次遍历同时计算 max 和 sum
    float thread_max = -FLT_MAX;
    float thread_sum = 0.0f;

    for (int col = tid; col < num_cols; col += block_size) {
        float val = row_input[col];
        if (val > thread_max) {
            // 关键: 修正之前的 sum
            // old_sum 中的所有 exp 值都是基于 old_max 计算的
            // 需要乘以 exp(old_max - new_max) 来修正
            thread_sum = thread_sum * expf(thread_max - val);
            thread_max = val;
        }
        thread_sum += expf(val - thread_max);
    }

    // Warp-level Online Reduction
    // 需要同时 reduce max 和 sum，且在 reduce max 时修正 sum
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float other_max = __shfl_down_sync(0xffffffff, thread_max, offset);
        float other_sum = __shfl_down_sync(0xffffffff, thread_sum, offset);

        if (other_max > thread_max) {
            thread_sum = thread_sum * expf(thread_max - other_max) + other_sum;
            thread_max = other_max;
        } else {
            thread_sum = thread_sum + other_sum * expf(other_max - thread_max);
        }
    }

    // Block-level reduction
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int num_warps = (block_size + WARP_SIZE - 1) / WARP_SIZE;

    // 存储每个 warp 的 (max, sum) 对
    if (lane_id == 0) {
        shared_data[warp_id * 2] = thread_max;
        shared_data[warp_id * 2 + 1] = thread_sum;
    }
    __syncthreads();

    if (warp_id == 0 && lane_id < num_warps) {
        thread_max = shared_data[lane_id * 2];
        thread_sum = shared_data[lane_id * 2 + 1];
    } else if (warp_id == 0) {
        thread_max = -FLT_MAX;
        thread_sum = 0.0f;
    }

    if (warp_id == 0) {
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            float other_max = __shfl_down_sync(0xffffffff, thread_max, offset);
            float other_sum = __shfl_down_sync(0xffffffff, thread_sum, offset);
            if (other_max > thread_max) {
                thread_sum = thread_sum * expf(thread_max - other_max) + other_sum;
                thread_max = other_max;
            } else {
                thread_sum = thread_sum + other_sum * expf(other_max - thread_max);
            }
        }
        if (lane_id == 0) {
            shared_data[0] = thread_max;
            shared_data[1] = thread_sum;
        }
    }
    __syncthreads();

    float row_max = shared_data[0];
    float row_sum = shared_data[1];

    // 归一化 (第二次遍历)
    for (int col = tid; col < num_cols; col += block_size) {
        row_output[col] = expf(row_input[col] - row_max) / row_sum;
    }
}

// ============================================
// CPU 参考实现
// ============================================
void softmax_cpu(const float* input, float* output, int num_rows, int num_cols) {
    for (int row = 0; row < num_rows; row++) {
        const float* row_in = input + row * num_cols;
        float* row_out = output + row * num_cols;

        float max_val = -FLT_MAX;
        for (int col = 0; col < num_cols; col++) {
            if (row_in[col] > max_val) max_val = row_in[col];
        }

        float sum = 0.0f;
        for (int col = 0; col < num_cols; col++) {
            row_out[col] = expf(row_in[col] - max_val);
            sum += row_out[col];
        }

        for (int col = 0; col < num_cols; col++) {
            row_out[col] /= sum;
        }
    }
}

// ============================================
// Main
// ============================================
int main() {
    printf("============================================================\n");
    printf("  CUDA 算子: Softmax (Naive + Online Softmax)\n");
    printf("============================================================\n");

    srand(42);
    const int num_rows = 1024, num_cols = 4096;
    size_t size = num_rows * num_cols * sizeof(float);

    float *h_input = (float*)malloc(size);
    float *h_output_ref = (float*)malloc(size);
    float *h_output_gpu = (float*)malloc(size);

    for (int i = 0; i < num_rows * num_cols; i++) {
        h_input[i] = (float)(rand() % 1000) / 100.0f - 5.0f;
    }

    softmax_cpu(h_input, h_output_ref, num_rows, num_cols);

    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, size));
    CUDA_CHECK(cudaMalloc(&d_output, size));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    int block_size = 256;
    int shared_mem_size = (block_size / WARP_SIZE) * 2 * sizeof(float);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    int warmup = 10, repeat = 100;
    float ms;

    // --- V1: Naive Safe Softmax ---
    for (int i = 0; i < warmup; i++)
        softmax_naive<<<num_rows, block_size, shared_mem_size>>>(d_input, d_output, num_rows, num_cols);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < repeat; i++)
        softmax_naive<<<num_rows, block_size, shared_mem_size>>>(d_input, d_output, num_rows, num_cols);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);

    CUDA_CHECK(cudaMemcpy(h_output_gpu, d_output, size, cudaMemcpyDeviceToHost));
    float diff = 0;
    for (int i = 0; i < num_rows * num_cols; i++)
        diff = fmaxf(diff, fabsf(h_output_ref[i] - h_output_gpu[i]));

    printf("\n  V1 Naive Safe Softmax:\n");
    printf("    时间: %.3f ms | 误差: %.2e %s\n", ms / repeat, diff, diff < 1e-5 ? "✅" : "❌");

    // --- V2: Online Softmax ---
    for (int i = 0; i < warmup; i++)
        softmax_online<<<num_rows, block_size, shared_mem_size>>>(d_input, d_output, num_rows, num_cols);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < repeat; i++)
        softmax_online<<<num_rows, block_size, shared_mem_size>>>(d_input, d_output, num_rows, num_cols);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);

    CUDA_CHECK(cudaMemcpy(h_output_gpu, d_output, size, cudaMemcpyDeviceToHost));
    diff = 0;
    for (int i = 0; i < num_rows * num_cols; i++)
        diff = fmaxf(diff, fabsf(h_output_ref[i] - h_output_gpu[i]));

    printf("\n  V2 Online Softmax:\n");
    printf("    时间: %.3f ms | 误差: %.2e %s\n", ms / repeat, diff, diff < 1e-5 ? "✅" : "❌");

    printf("\n  💡 面试要点:\n");
    printf("    1. Safe Softmax: 减 max 防止 exp 溢出 (e^{700} = inf)\n");
    printf("    2. Online Softmax: 遇到新 max 时修正 sum = sum * exp(old_max - new_max)\n");
    printf("    3. Warp Reduction: __shfl_down_sync 比 Shared Memory 更快\n");
    printf("    4. Online Softmax 是 FlashAttention 分块计算的数学基础\n");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    free(h_input);
    free(h_output_ref);
    free(h_output_gpu);

    return 0;
}
