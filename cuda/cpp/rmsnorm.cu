/*
 * CUDA 算子: RMSNorm (Root Mean Square Normalization)
 * ===================================================
 * Qwen2.5 / LLaMA 使用的归一化层。
 *
 * 面试考点:
 * ---------
 * 1. RMSNorm vs LayerNorm: 去掉 mean，只保留 variance (用 x^2 的均值)
 * 2. Warp Reduction 求 sum(x^2)
 * 3. rsqrt 比 sqrt + div 更快 (一条指令)
 * 4. 混合精度: 输入 fp16，用 fp32 累加，输出 fp16
 * 5. 为什么是 memory-bound: 计算量 O(d)，访存量 O(d)，算术强度 ~1
 *
 * 编译运行:
 *   nvcc -O3 -arch=sm_89 rmsnorm.cu -o rmsnorm && ./rmsnorm
 *
 * 公式:
 *   y = x / sqrt(mean(x^2) + eps) * weight
 *     = x * rsqrt(sum(x^2) / d + eps) * weight
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
// Version 1: 基础版 RMSNorm (每行一个 block)
// ============================================
/*
 * 每个 block 处理一行 (一个 token 的 hidden_state)
 * 
 * 步骤:
 * 1. 每个线程计算部分 x^2 的和
 * 2. Warp Reduction + Block Reduction 求 sum(x^2)
 * 3. 计算 rsqrt(mean + eps)
 * 4. 归一化并乘以 weight
 */
__global__ void rmsnorm_v1(
    const float* __restrict__ input,    // [num_rows, hidden_size]
    const float* __restrict__ weight,   // [hidden_size]
    float* __restrict__ output,         // [num_rows, hidden_size]
    int num_rows, int hidden_size, float epsilon
) {
    extern __shared__ float shared_data[];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    if (row >= num_rows) return;

    const float* row_input = input + row * hidden_size;
    float* row_output = output + row * hidden_size;

    // Step 1: 每个线程计算部分 sum(x^2)
    float thread_sum_sq = 0.0f;
    for (int col = tid; col < hidden_size; col += block_size) {
        float val = row_input[col];
        thread_sum_sq += val * val;
    }

    // Step 2: Warp-level reduction
    thread_sum_sq = warp_reduce_sum(thread_sum_sq);

    // Block-level reduction via shared memory
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int num_warps = (block_size + WARP_SIZE - 1) / WARP_SIZE;

    if (lane_id == 0) {
        shared_data[warp_id] = thread_sum_sq;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < num_warps) ? shared_data[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane_id == 0) {
            // Step 3: 计算 rsqrt(mean(x^2) + eps)
            // rsqrt = 1 / sqrt(x)，是一条 GPU 指令，比 sqrt + div 快
            shared_data[0] = rsqrtf(val / (float)hidden_size + epsilon);
        }
    }
    __syncthreads();

    float inv_rms = shared_data[0];

    // Step 4: 归一化并乘以 weight
    for (int col = tid; col < hidden_size; col += block_size) {
        row_output[col] = row_input[col] * inv_rms * weight[col];
    }
}

// ============================================
// Version 2: 向量化访存 (float4)
// ============================================
/*
 * 使用 float4 一次加载 4 个 float (128 bits)
 * 
 * 优势:
 * - 单次内存事务加载 128 bits vs 32 bits
 * - 更好地利用内存带宽
 * - 减少内存事务数量
 *
 * 要求: hidden_size 必须是 4 的倍数 (LLM 中通常满足)
 */
__global__ void rmsnorm_vectorized(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int num_rows, int hidden_size, float epsilon
) {
    extern __shared__ float shared_data[];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    if (row >= num_rows) return;

    // 向量化指针
    int vec_hidden_size = hidden_size / 4;
    const float4* row_input_vec = reinterpret_cast<const float4*>(input + row * hidden_size);
    const float4* weight_vec = reinterpret_cast<const float4*>(weight);
    float4* row_output_vec = reinterpret_cast<float4*>(output + row * hidden_size);

    // Step 1: 向量化计算 sum(x^2)
    float thread_sum_sq = 0.0f;
    for (int col = tid; col < vec_hidden_size; col += block_size) {
        float4 val = row_input_vec[col];
        thread_sum_sq += val.x * val.x + val.y * val.y + val.z * val.z + val.w * val.w;
    }

    // Step 2: Reduction
    thread_sum_sq = warp_reduce_sum(thread_sum_sq);

    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int num_warps = (block_size + WARP_SIZE - 1) / WARP_SIZE;

    if (lane_id == 0) shared_data[warp_id] = thread_sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < num_warps) ? shared_data[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane_id == 0) {
            shared_data[0] = rsqrtf(val / (float)hidden_size + epsilon);
        }
    }
    __syncthreads();

    float inv_rms = shared_data[0];

    // Step 3: 向量化归一化
    for (int col = tid; col < vec_hidden_size; col += block_size) {
        float4 val = row_input_vec[col];
        float4 w = weight_vec[col];
        float4 result;
        result.x = val.x * inv_rms * w.x;
        result.y = val.y * inv_rms * w.y;
        result.z = val.z * inv_rms * w.z;
        result.w = val.w * inv_rms * w.w;
        row_output_vec[col] = result;
    }
}

// ============================================
// CPU 参考实现
// ============================================
void rmsnorm_cpu(
    const float* input, const float* weight, float* output,
    int num_rows, int hidden_size, float epsilon
) {
    for (int row = 0; row < num_rows; row++) {
        const float* row_in = input + row * hidden_size;
        float* row_out = output + row * hidden_size;

        float sum_sq = 0.0f;
        for (int col = 0; col < hidden_size; col++) {
            sum_sq += row_in[col] * row_in[col];
        }

        float inv_rms = 1.0f / sqrtf(sum_sq / hidden_size + epsilon);

        for (int col = 0; col < hidden_size; col++) {
            row_out[col] = row_in[col] * inv_rms * weight[col];
        }
    }
}

// ============================================
// Main
// ============================================
int main() {
    printf("============================================================\n");
    printf("  CUDA 算子: RMSNorm (基础版 + 向量化版)\n");
    printf("============================================================\n");

    srand(42);

    // 模拟 Qwen2.5-7B: hidden_size = 4096
    const int num_rows = 2048;  // batch * seq_len
    const int hidden_size = 4096;
    const float epsilon = 1e-6f;

    size_t input_size = num_rows * hidden_size * sizeof(float);
    size_t weight_size = hidden_size * sizeof(float);

    float *h_input = (float*)malloc(input_size);
    float *h_weight = (float*)malloc(weight_size);
    float *h_output_ref = (float*)malloc(input_size);
    float *h_output_gpu = (float*)malloc(input_size);

    for (int i = 0; i < num_rows * hidden_size; i++)
        h_input[i] = (float)(rand() % 200 - 100) / 100.0f;
    for (int i = 0; i < hidden_size; i++)
        h_weight[i] = 1.0f;

    rmsnorm_cpu(h_input, h_weight, h_output_ref, num_rows, hidden_size, epsilon);

    float *d_input, *d_weight, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, input_size));
    CUDA_CHECK(cudaMalloc(&d_weight, weight_size));
    CUDA_CHECK(cudaMalloc(&d_output, input_size));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weight, h_weight, weight_size, cudaMemcpyHostToDevice));

    int block_size = 256;
    int shared_mem = (block_size / WARP_SIZE) * sizeof(float);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    int warmup = 10, repeat = 100;
    float ms, diff;

    // --- V1: 基础版 ---
    for (int i = 0; i < warmup; i++)
        rmsnorm_v1<<<num_rows, block_size, shared_mem>>>(d_input, d_weight, d_output, num_rows, hidden_size, epsilon);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < repeat; i++)
        rmsnorm_v1<<<num_rows, block_size, shared_mem>>>(d_input, d_weight, d_output, num_rows, hidden_size, epsilon);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);

    CUDA_CHECK(cudaMemcpy(h_output_gpu, d_output, input_size, cudaMemcpyDeviceToHost));
    diff = 0;
    for (int i = 0; i < num_rows * hidden_size; i++)
        diff = fmaxf(diff, fabsf(h_output_ref[i] - h_output_gpu[i]));

    printf("\n  V1 基础版 RMSNorm:\n");
    printf("    时间: %.3f ms | 误差: %.2e %s\n", ms / repeat, diff, diff < 1e-4 ? "✅" : "❌");

    // --- V2: 向量化版 ---
    for (int i = 0; i < warmup; i++)
        rmsnorm_vectorized<<<num_rows, block_size, shared_mem>>>(d_input, d_weight, d_output, num_rows, hidden_size, epsilon);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < repeat; i++)
        rmsnorm_vectorized<<<num_rows, block_size, shared_mem>>>(d_input, d_weight, d_output, num_rows, hidden_size, epsilon);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);

    CUDA_CHECK(cudaMemcpy(h_output_gpu, d_output, input_size, cudaMemcpyDeviceToHost));
    diff = 0;
    for (int i = 0; i < num_rows * hidden_size; i++)
        diff = fmaxf(diff, fabsf(h_output_ref[i] - h_output_gpu[i]));

    printf("\n  V2 向量化版 RMSNorm (float4):\n");
    printf("    时间: %.3f ms | 误差: %.2e %s\n", ms / repeat, diff, diff < 1e-4 ? "✅" : "❌");

    printf("\n  💡 面试要点:\n");
    printf("    1. RMSNorm 是 memory-bound 算子: 算术强度 ~1 FLOPs/byte\n");
    printf("    2. rsqrt 是一条 GPU 指令，比 1/sqrt(x) 快\n");
    printf("    3. float4 向量化访存: 单次事务 128 bits，提升带宽利用率\n");
    printf("    4. 混合精度: 输入 fp16，用 fp32 累加防止精度损失\n");
    printf("    5. 适合与 Residual Add 融合 (见 fused_add_rmsnorm)\n");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_weight));
    CUDA_CHECK(cudaFree(d_output));
    free(h_input);
    free(h_weight);
    free(h_output_ref);
    free(h_output_gpu);

    return 0;
}
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < repeat; i++)
        rmsnorm_vectorized<<<num_rows, block_size, shared_mem>>>(d_input, d_weight, d_output, num_rows, hidden_size, epsilon);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);

    CUDA_CHECK(cudaMemcpy(h_output_gpu, d_output, input_size, cudaMemcpyDeviceToHost));
    diff = 0;
    for (int i = 0; i