/*
 * CUDA 算子: GEMV (General Matrix-Vector Multiplication)
 * ======================================================
 * GEMV 是 LLM 推理 decode 阶段的核心算子。
 * 当 batch_size=1 时，GEMM 退化为 GEMV，此时优化策略完全不同。
 *
 * 面试考点:
 * ---------
 * 1. LLM 推理中 prefill vs decode 的计算特征差异
 *    - Prefill: 大矩阵乘法 (GEMM)，compute-bound
 *    - Decode:  矩阵-向量乘法 (GEMV)，memory-bound
 * 2. GEMV 为什么是 memory-bound: 矩阵只用一次，无法复用
 * 3. 行并行 vs 列并行策略
 * 4. Warp-level reduce 在 GEMV 中的应用
 * 5. 向量化访存对 GEMV 的加速效果
 *
 * 编译运行:
 *   nvcc -O3 -arch=sm_89 gemv.cu -o gemv && ./gemv
 *
 * 计算:
 *   y[M] = A[M, K] @ x[K]
 *   y[i] = sum_k(A[i][k] * x[k])
 *
 * 为什么 GEMV 是 memory-bound:
 *   - 矩阵 A 的每个元素只被读取一次 (不像 GEMM 可以在 shared memory 中复用)
 *   - 计算量: 2*M*K FLOPs
 *   - 访存量: (M*K + K + M) * 4 bytes ≈ M*K*4 bytes
 *   - 算术强度: 2*M*K / (M*K*4) = 0.5 FLOPs/byte (极低)
 */

#include <stdio.h>
#include <stdlib.h>
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

__device__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// ============================================
// Version 1: 行并行 (每个线程计算一行的点积)
// ============================================
/*
 * 最简单的策略: 每个线程负责 y 的一个元素。
 * 
 * 优点: 简单直观
 * 缺点: 
 *   - 每个线程串行遍历 K 个元素
 *   - 当 K 很大时 (如 4096)，单线程工作量大
 *   - x 向量被所有线程重复读取 (可以用 shared memory 缓存)
 */
__global__ void gemv_row_parallel(
    const float* __restrict__ A,  // [M, K]
    const float* __restrict__ x,  // [K]
    float* __restrict__ y,        // [M]
    int M, int K
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * x[k];
        }
        y[row] = sum;
    }
}

// ============================================
// Version 2: Warp 协作 (每个 warp 计算一行)
// ============================================
/*
 * 一个 warp (32 threads) 协作计算 y 的一个元素:
 *   - 32 个线程分别处理 K/32 个元素
 *   - 用 warp shuffle reduce 求和
 *
 * 优点:
 *   - 更好的内存访问模式 (32 个线程连续访问 A 的一行)
 *   - Warp shuffle reduce 无需 shared memory
 *   - 适合 K 较大的情况 (如 K=4096)
 */
__global__ void gemv_warp_per_row(
    const float* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    if (warp_id >= M) return;

    const float* row_A = A + warp_id * K;

    // 每个线程处理 K/32 个元素
    float thread_sum = 0.0f;
    for (int k = lane_id; k < K; k += WARP_SIZE) {
        thread_sum += row_A[k] * x[k];
    }

    // Warp-level reduce
    thread_sum = warp_reduce_sum(thread_sum);

    // Lane 0 写结果
    if (lane_id == 0) {
        y[warp_id] = thread_sum;
    }
}

// ============================================
// Version 3: Warp 协作 + 向量化访存 (float4)
// ============================================
/*
 * 在 V2 基础上使用 float4 向量化加载:
 * - 每次加载 4 个 float (128 bits)
 * - 减少内存事务数量
 * - 对 memory-bound 的 GEMV 效果显著
 *
 * 要求: K 必须是 4 的倍数 (LLM 中 hidden_size 通常满足)
 */
__global__ void gemv_warp_vectorized(
    const float* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    if (warp_id >= M) return;

    const float4* row_A_vec = reinterpret_cast<const float4*>(A + warp_id * K);
    const float4* x_vec = reinterpret_cast<const float4*>(x);
    int K_vec = K / 4;

    float thread_sum = 0.0f;
    for (int k = lane_id; k < K_vec; k += WARP_SIZE) {
        float4 a_val = row_A_vec[k];
        float4 x_val = x_vec[k];
        thread_sum += a_val.x * x_val.x + a_val.y * x_val.y +
                      a_val.z * x_val.z + a_val.w * x_val.w;
    }

    thread_sum = warp_reduce_sum(thread_sum);

    if (lane_id == 0) {
        y[warp_id] = thread_sum;
    }
}

// ============================================
// Version 4: Block 协作 + Shared Memory 缓存 x
// ============================================
/*
 * 当多行共享同一个 x 向量时，可以将 x 缓存到 shared memory:
 * - 一个 block 处理多行 (ROWS_PER_BLOCK 行)
 * - x 向量分块加载到 shared memory，所有行共享
 * - 减少 x 的 global memory 访问次数
 *
 * 适合 M 较大、K 适中的场景。
 */
#define ROWS_PER_BLOCK 4
#define TILE_K 256

__global__ void gemv_block_shared(
    const float* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    __shared__ float x_shared[TILE_K];

    int block_row_start = blockIdx.x * ROWS_PER_BLOCK;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    // 每个线程维护 ROWS_PER_BLOCK 个局部和
    float sums[ROWS_PER_BLOCK] = {0.0f};

    // 沿 K 维度分块
    for (int tile_start = 0; tile_start < K; tile_start += TILE_K) {
        // 协作加载 x 的一个 tile 到 shared memory
        int tile_end = min(tile_start + TILE_K, K);
        int tile_len = tile_end - tile_start;

        for (int i = tid; i < tile_len; i += block_size) {
            x_shared[i] = x[tile_start + i];
        }
        __syncthreads();

        // 每个线程处理多行
        for (int r = 0; r < ROWS_PER_BLOCK; r++) {
            int row = block_row_start + r;
            if (row >= M) break;

            const float* row_A = A + row * K + tile_start;
            for (int k = tid; k < tile_len; k += block_size) {
                sums[r] += row_A[k] * x_shared[k];
            }
        }
        __syncthreads();
    }

    // Reduce 每行的部分和
    for (int r = 0; r < ROWS_PER_BLOCK; r++) {
        int row = block_row_start + r;
        if (row >= M) break;

        float val = warp_reduce_sum(sums[r]);

        int warp_id = tid / WARP_SIZE;
        int lane_id = tid % WARP_SIZE;

        __shared__ float reduce_shared[32];
        if (lane_id == 0) reduce_shared[warp_id] = val;
        __syncthreads();

        if (warp_id == 0) {
            int num_warps = block_size / WARP_SIZE;
            val = (lane_id < num_warps) ? reduce_shared[lane_id] : 0.0f;
            val = warp_reduce_sum(val);
            if (lane_id == 0) y[row] = val;
        }
        __syncthreads();
    }
}

// ============================================
// CPU 参考实现
// ============================================
void gemv_cpu(const float* A, const float* x, float* y, int M, int K) {
    for (int i = 0; i < M; i++) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[i * K + k] * x[k];
        }
        y[i] = sum;
    }
}

// ============================================
// Main
// ============================================
int main() {
    printf("============================================================\n");
    printf("  CUDA 算子: GEMV (矩阵-向量乘法, LLM Decode 核心)\n");
    printf("============================================================\n");

    srand(42);

    // 模拟 Qwen2.5-7B 的线性层: hidden_size=4096
    const int M = 4096;  // 输出维度
    const int K = 4096;  // 输入维度 (hidden_size)

    size_t size_A = M * K * sizeof(float);
    size_t size_x = K * sizeof(float);
    size_t size_y = M * sizeof(float);

    float *h_A = (float*)malloc(size_A);
    float *h_x = (float*)malloc(size_x);
    float *h_y_ref = (float*)malloc(size_y);
    float *h_y_gpu = (float*)malloc(size_y);

    for (int i = 0; i < M * K; i++) h_A[i] = (float)(rand() % 200 - 100) / 1000.0f;
    for (int i = 0; i < K; i++) h_x[i] = (float)(rand() % 200 - 100) / 100.0f;

    gemv_cpu(h_A, h_x, h_y_ref, M, K);

    float *d_A, *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_x, size_x));
    CUDA_CHECK(cudaMalloc(&d_y, size_y));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, size_x, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    int warmup = 20, repeat = 200;
    float ms, diff;
    float gflops = 2.0f * M * K / 1e9;

    // --- V1: 行并行 ---
    {
        int block_size = 256;
        int grid_size = (M + block_size - 1) / block_size;

        for (int i = 0; i < warmup; i++)
            gemv_row_parallel<<<grid_size, block_size>>>(d_A, d_x, d_y, M, K);
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        for (int i = 0; i < repeat; i++)
            gemv_row_parallel<<<grid_size, block_size>>>(d_A, d_x, d_y, M, K);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);

        CUDA_CHECK(cudaMemcpy(h_y_gpu, d_y, size_y, cudaMemcpyDeviceToHost));
        diff = 0;
        for (int i = 0; i < M; i++) diff = fmaxf(diff, fabsf(h_y_ref[i] - h_y_gpu[i]));

        float bandwidth = (size_A + size_x + size_y) / (ms / repeat / 1000.0f) / 1e9;
        printf("\n  V1 行并行 (每线程一行):\n");
        printf("    时间: %.3f ms | GFLOPS: %.1f | 带宽: %.1f GB/s | 误差: %.2e %s\n",
               ms / repeat, gflops / (ms / repeat) * 1000, bandwidth, diff, diff < 1e-1 ? "✅" : "❌");
    }

    // --- V2: Warp 协作 ---
    {
        int warps_needed = M;
        int threads_needed = warps_needed * WARP_SIZE;
        int block_size = 256;
        int grid_size = (threads_needed + block_size - 1) / block_size;

        for (int i = 0; i < warmup; i++)
            gemv_warp_per_row<<<grid_size, block_size>>>(d_A, d_x, d_y, M, K);
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        for (int i = 0; i < repeat; i++)
            gemv_warp_per_row<<<grid_size, block_size>>>(d_A, d_x, d_y, M, K);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);

        CUDA_CHECK(cudaMemcpy(h_y_gpu, d_y, size_y, cudaMemcpyDeviceToHost));
        diff = 0;
        for (int i = 0; i < M; i++) diff = fmaxf(diff, fabsf(h_y_ref[i] - h_y_gpu[i]));

        float bandwidth = (size_A + size_x + size_y) / (ms / repeat / 1000.0f) / 1e9;
        printf("\n  V2 Warp 协作 (每 warp 一行):\n");
        printf("    时间: %.3f ms | GFLOPS: %.1f | 带宽: %.1f GB/s | 误差: %.2e %s\n",
               ms / repeat, gflops / (ms / repeat) * 1000, bandwidth, diff, diff < 1e-1 ? "✅" : "❌");
    }

    // --- V3: Warp + 向量化 ---
    {
        int warps_needed = M;
        int threads_needed = warps_needed * WARP_SIZE;
        int block_size = 256;
        int grid_size = (threads_needed + block_size - 1) / block_size;

        for (int i = 0; i < warmup; i++)
            gemv_warp_vectorized<<<grid_size, block_size>>>(d_A, d_x, d_y, M, K);
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        for (int i = 0; i < repeat; i++)
            gemv_warp_vectorized<<<grid_size, block_size>>>(d_A, d_x, d_y, M, K);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);

        CUDA_CHECK(cudaMemcpy(h_y_gpu, d_y, size_y, cudaMemcpyDeviceToHost));
        diff = 0;
        for (int i = 0; i < M; i++) diff = fmaxf(diff, fabsf(h_y_ref[i] - h_y_gpu[i]));

        float bandwidth = (size_A + size_x + size_y) / (ms / repeat / 1000.0f) / 1e9;
        printf("\n  V3 Warp + Vectorized (float4):\n");
        printf("    时间: %.3f ms | GFLOPS: %.1f | 带宽: %.1f GB/s | 误差: %.2e %s\n",
               ms / repeat, gflops / (ms / repeat) * 1000, bandwidth, diff, diff < 1e-1 ? "✅" : "❌");
    }

    // --- V4: Block + Shared Memory ---
    {
        int grid_size = (M + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
        int block_size = 256;

        for (int i = 0; i < warmup; i++)
            gemv_block_shared<<<grid_size, block_size>>>(d_A, d_x, d_y, M, K);
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        for (int i = 0; i < repeat; i++)
            gemv_block_shared<<<grid_size, block_size>>>(d_A, d_x, d_y, M, K);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);

        CUDA_CHECK(cudaMemcpy(h_y_gpu, d_y, size_y, cudaMemcpyDeviceToHost));
        diff = 0;
        for (int i = 0; i < M; i++) diff = fmaxf(diff, fabsf(h_y_ref[i] - h_y_gpu[i]));

        float bandwidth = (size_A + size_x + size_y) / (ms / repeat / 1000.0f) / 1e9;
        printf("\n  V4 Block + Shared Memory (缓存 x):\n");
        printf("    时间: %.3f ms | GFLOPS: %.1f | 带宽: %.1f GB/s | 误差: %.2e %s\n",
               ms / repeat, gflops / (ms / repeat) * 1000, bandwidth, diff, diff < 1e-1 ? "✅" : "❌");
    }

    printf("\n  💡 面试要点:\n");
    printf("    1. LLM Decode 阶段: batch=1, GEMM 退化为 GEMV, memory-bound\n");
    printf("    2. GEMV 算术强度 ~0.5 FLOPs/byte, 远低于 GPU 拐点\n");
    printf("    3. 优化方向: 提升带宽利用率 (向量化, 合并访存)\n");
    printf("    4. Warp 协作: 32 线程共同计算一行点积, warp shuffle reduce\n");
    printf("    5. Shared Memory 缓存 x: 多行共享, 减少重复读取\n");
    printf("    6. 这就是为什么 LLM 推理要做 batching: 把 GEMV 变回 GEMM!\n");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    free(h_A);
    free(h_x);
    free(h_y_ref);
    free(h_y_gpu);

    return 0;
}
