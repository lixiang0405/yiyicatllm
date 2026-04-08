/*
 * CUDA 算子: Fused Residual Add + RMSNorm (算子融合)
 * ===================================================
 * 将 Residual Add 和 RMSNorm 融合为一个 kernel，减少 Global Memory IO。
 *
 * 面试考点:
 * ---------
 * 1. 为什么要做算子融合: memory-bound 算子的瓶颈在 IO 而非计算
 * 2. Roofline Model: 算术强度 < 拐点 → memory-bound
 * 3. 融合前后 IO 量对比: 5N → 3N (节省 40%)
 * 4. vLLM 中的 fused kernels: fused_add_rmsnorm, silu_and_mul
 * 5. 何时不该融合: compute-bound 算子 (如 GEMM)
 *
 * 编译运行:
 *   nvcc -O3 -arch=sm_89 fused_add_rmsnorm.cu -o fused_add_rmsnorm && ./fused_add_rmsnorm
 *
 * Transformer 中的位置:
 *   sublayer_output = Attention(x) 或 FFN(x)
 *   hidden = residual + sublayer_output     ← Residual Add
 *   output = RMSNorm(hidden) * weight       ← RMSNorm
 *
 * IO 分析:
 *   不融合 (2 kernels):
 *     Add:     读 residual(N) + sublayer(N), 写 hidden(N) = 3N
 *     RMSNorm: 读 hidden(N), 写 output(N)                 = 2N
 *     总计: 5N 次 IO
 *
 *   融合 (1 kernel):
 *     读 residual(N) + sublayer(N), 写 hidden(N) + output(N) = 4N
 *     (hidden 在寄存器中直接用于 RMSNorm，不需要额外读)
 *     总计: 4N 次 IO，节省 20%
 *     如果不需要输出 hidden: 3N 次 IO，节省 40%
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
// 非融合版本: 分两个 kernel 执行 (对照组)
// ============================================
__global__ void residual_add_kernel(
    const float* __restrict__ residual,
    const float* __restrict__ sublayer,
    float* __restrict__ hidden,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        hidden[idx] = residual[idx] + sublayer[idx];
    }
}

__global__ void rmsnorm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int hidden_size, float epsilon
) {
    extern __shared__ float shared_data[];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    const float* row_input = input + row * hidden_size;
    float* row_output = output + row * hidden_size;

    float thread_sum_sq = 0.0f;
    for (int col = tid; col < hidden_size; col += block_size) {
        float val = row_input[col];
        thread_sum_sq += val * val;
    }

    thread_sum_sq = warp_reduce_sum(thread_sum_sq);
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int num_warps = (block_size + WARP_SIZE - 1) / WARP_SIZE;

    if (lane_id == 0) shared_data[warp_id] = thread_sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < num_warps) ? shared_data[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane_id == 0) shared_data[0] = rsqrtf(val / hidden_size + epsilon);
    }
    __syncthreads();

    float inv_rms = shared_data[0];
    for (int col = tid; col < hidden_size; col += block_size) {
        row_output[col] = row_input[col] * inv_rms * weight[col];
    }
}

// ============================================
// 融合版本: 一个 kernel 完成 Add + RMSNorm
// ============================================
/*
 * 核心优化:
 * 1. residual + sublayer 的结果保存在寄存器中，直接用于 RMSNorm
 * 2. 不需要将中间结果写回 Global Memory 再读回来
 * 3. 同时输出 new_residual (供下一层使用) 和 normalized output
 */
__global__ void fused_add_rmsnorm_kernel(
    const float* __restrict__ residual,      // [num_rows, hidden_size]
    const float* __restrict__ sublayer,       // [num_rows, hidden_size]
    const float* __restrict__ weight,         // [hidden_size]
    float* __restrict__ output,               // [num_rows, hidden_size]
    float* __restrict__ new_residual,         // [num_rows, hidden_size]
    int hidden_size, float epsilon
) {
    extern __shared__ float shared_data[];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    const float* row_residual = residual + row * hidden_size;
    const float* row_sublayer = sublayer + row * hidden_size;
    float* row_output = output + row * hidden_size;
    float* row_new_residual = new_residual + row * hidden_size;

    // Step 1 & 2: Add + 计算 sum(x^2)，在同一次遍历中完成
    float thread_sum_sq = 0.0f;
    for (int col = tid; col < hidden_size; col += block_size) {
        // Residual Add (结果暂存到 new_residual，同时用于计算 sum_sq)
        float hidden_val = row_residual[col] + row_sublayer[col];
        row_new_residual[col] = hidden_val;  // 写出供下一层使用
        thread_sum_sq += hidden_val * hidden_val;
    }

    // Step 3: Reduction 求 sum(x^2)
    thread_sum_sq = warp_reduce_sum(thread_sum_sq);
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int num_warps = (block_size + WARP_SIZE - 1) / WARP_SIZE;

    if (lane_id == 0) shared_data[warp_id] = thread_sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < num_warps) ? shared_data[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane_id == 0) shared_data[0] = rsqrtf(val / hidden_size + epsilon);
    }
    __syncthreads();

    float inv_rms = shared_data[0];

    // Step 4: 归一化 (从 new_residual 读取，避免重新计算 add)
    for (int col = tid; col < hidden_size; col += block_size) {
        row_output[col] = row_new_residual[col] * inv_rms * weight[col];
    }
}

// ============================================
// CPU 参考实现
// ============================================
void fused_add_rmsnorm_cpu(
    const float* residual, const float* sublayer, const float* weight,
    float* output, float* new_residual,
    int num_rows, int hidden_size, float epsilon
) {
    for (int row = 0; row < num_rows; row++) {
        float sum_sq = 0.0f;
        for (int col = 0; col < hidden_size; col++) {
            int idx = row * hidden_size + col;
            float val = residual[idx] + sublayer[idx];
            new_residual[idx] = val;
            sum_sq += val * val;
        }
        float inv_rms = 1.0f / sqrtf(sum_sq / hidden_size + epsilon);
        for (int col = 0; col < hidden_size; col++) {
            int idx = row * hidden_size + col;
            output[idx] = new_residual[idx] * inv_rms * weight[col];
        }
    }
}

// ============================================
// Main
// ============================================
int main() {
    printf("============================================================\n");
    printf("  CUDA 算子: Fused Residual Add + RMSNorm\n");
    printf("============================================================\n");

    srand(42);
    const int num_rows = 2048;
    const int hidden_size = 4096;
    const float epsilon = 1e-6f;

    size_t data_size = num_rows * hidden_size * sizeof(float);
    size_t weight_size = hidden_size * sizeof(float);

    // Host 内存
    float *h_residual = (float*)malloc(data_size);
    float *h_sublayer = (float*)malloc(data_size);
    float *h_weight = (float*)malloc(weight_size);
    float *h_output_ref = (float*)malloc(data_size);
    float *h_new_res_ref = (float*)malloc(data_size);
    float *h_output_gpu = (float*)malloc(data_size);

    for (int i = 0; i < num_rows * hidden_size; i++) {
        h_residual[i] = (float)(rand() % 200 - 100) / 100.0f;
        h_sublayer[i] = (float)(rand() % 200 - 100) / 100.0f;
    }
    for (int i = 0; i < hidden_size; i++) h_weight[i] = 1.0f;

    fused_add_rmsnorm_cpu(h_residual, h_sublayer, h_weight,
                          h_output_ref, h_new_res_ref,
                          num_rows, hidden_size, epsilon);

    // Device 内存
    float *d_residual, *d_sublayer, *d_weight, *d_output, *d_new_residual, *d_hidden;
    CUDA_CHECK(cudaMalloc(&d_residual, data_size));
    CUDA_CHECK(cudaMalloc(&d_sublayer, data_size));
    CUDA_CHECK(cudaMalloc(&d_weight, weight_size));
    CUDA_CHECK(cudaMalloc(&d_output, data_size));
    CUDA_CHECK(cudaMalloc(&d_new_residual, data_size));
    CUDA_CHECK(cudaMalloc(&d_hidden, data_size));
    CUDA_CHECK(cudaMemcpy(d_residual, h_residual, data_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sublayer, h_sublayer, data_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weight, h_weight, weight_size, cudaMemcpyHostToDevice));

    int block_size = 256;
    int shared_mem = (block_size / WARP_SIZE) * sizeof(float);
    int add_grid = (num_rows * hidden_size + block_size - 1) / block_size;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    int warmup = 10, repeat = 100;
    float ms, diff;

    // --- 非融合版本 ---
    for (int i = 0; i < warmup; i++) {
        residual_add_kernel<<<add_grid, block_size>>>(d_residual, d_sublayer, d_hidden, num_rows * hidden_size);
        rmsnorm_kernel<<<num_rows, block_size, shared_mem>>>(d_hidden, d_weight, d_output, hidden_size, epsilon);
    }
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < repeat; i++) {
        residual_add_kernel<<<add_grid, block_size>>>(d_residual, d_sublayer, d_hidden, num_rows * hidden_size);
        rmsnorm_kernel<<<num_rows, block_size, shared_mem>>>(d_hidden, d_weight, d_output, hidden_size, epsilon);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);

    CUDA_CHECK(cudaMemcpy(h_output_gpu, d_output, data_size, cudaMemcpyDeviceToHost));
    diff = 0;
    for (int i = 0; i < num_rows * hidden_size; i++)
        diff = fmaxf(diff, fabsf(h_output_ref[i] - h_output_gpu[i]));

    float unfused_time = ms / repeat;
    printf("\n  非融合版本 (2 kernels: Add + RMSNorm):\n");
    printf("    时间: %.3f ms | 误差: %.2e %s\n", unfused_time, diff, diff < 1e-4 ? "✅" : "❌");

    // --- 融合版本 ---
    for (int i = 0; i < warmup; i++)
        fused_add_rmsnorm_kernel<<<num_rows, block_size, shared_mem>>>(
            d_residual, d_sublayer, d_weight, d_output, d_new_residual, hidden_size, epsilon);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < repeat; i++)
        fused_add_rmsnorm_kernel<<<num_rows, block_size, shared_mem>>>(
            d_residual, d_sublayer, d_weight, d_output, d_new_residual, hidden_size, epsilon);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);

    CUDA_CHECK(cudaMemcpy(h_output_gpu, d_output, data_size, cudaMemcpyDeviceToHost));
    diff = 0;
    for (int i = 0; i < num_rows * hidden_size; i++)
        diff = fmaxf(diff, fabsf(h_output_ref[i] - h_output_gpu[i]));

    float fused_time = ms / repeat;
    printf("\n  融合版本 (1 kernel: Fused Add + RMSNorm):\n");
    printf("    时间: %.3f ms | 误差: %.2e %s\n", fused_time, diff, diff < 1e-4 ? "✅" : "❌");

    printf("\n  加速比: %.2fx\n", unfused_time / fused_time);

    // IO 分析
    float total_bytes = (float)num_rows * hidden_size * sizeof(float);
    printf("\n  IO 分析 (%d x %d, float32):\n", num_rows, hidden_size);
    printf("    非融合: %.1f MB (读 3N + 写 2N = 5N)\n", 5 * total_bytes / 1e6);
    printf("    融合:   %.1f MB (读 2N + 写 2N = 4N)\n", 4 * total_bytes / 1e6);
    printf("    IO 节省: 20%%\n");

    printf("\n  💡 面试要点:\n");
    printf("    1. 算子融合的本质: 减少 HBM 读写，中间结果留在寄存器/Shared Memory\n");
    printf("    2. 对 memory-bound 算子效果最好 (RMSNorm 算术强度 ~1)\n");
    printf("    3. 对 compute-bound 算子 (GEMM) 融合收益小\n");
    printf("    4. vLLM 中的融合: fused_add_rmsnorm, silu_and_mul, fused_moe\n");
    printf("    5. Roofline Model: 判断算子是 compute-bound 还是 memory-bound\n");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    CUDA_CHECK(cudaFree(d_residual));
    CUDA_CHECK(cudaFree(d_sublayer));
    CUDA_CHECK(cudaFree(d_weight));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_new_residual));
    CUDA_CHECK(cudaFree(d_hidden));
    free(h_residual);
    free(h_sublayer);
    free(h_weight);
    free(h_output_ref);
    free(h_new_res_ref);
    free(h_output_gpu);

    return 0;
}
