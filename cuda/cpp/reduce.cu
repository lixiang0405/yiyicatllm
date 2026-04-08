/*
 * CUDA 算子: Parallel Reduce (并行归约)
 * ======================================
 * Reduce 是所有 reduction 类算子的基础 (sum, max, min, mean)。
 * Softmax 中的 max/sum、RMSNorm 中的 sum(x^2) 都依赖 reduce。
 *
 * 面试考点:
 * ---------
 * 1. Naive reduce 的 warp divergence 问题
 * 2. 交错寻址 (Interleaved Addressing) 消除 bank conflict
 * 3. 循环展开 (Loop Unrolling) 减少同步开销
 * 4. Warp-level primitives: __shfl_down_sync (最快的 reduce)
 * 5. 向量化加载 (float4) 提升带宽利用率
 * 6. 两阶段 reduce: block-level → grid-level
 *
 * 编译运行:
 *   nvcc -O3 -arch=sm_89 reduce.cu -o reduce && ./reduce
 *
 * Reduce 的并行化思路:
 *   N 个元素 → 每个线程处理多个元素 → block 内 reduce → block 间 reduce
 *
 *   Step 1: 每个线程用 grid-stride loop 累加局部和
 *   Step 2: Warp 内用 __shfl_down_sync 归约 (32→1)
 *   Step 3: Warp 间用 Shared Memory 归约
 *   Step 4: 每个 block 输出一个部分和
 *   Step 5: 再用一个 kernel 归约所有 block 的部分和
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

// ============================================
// Version 1: Naive Reduce (Shared Memory, 交错寻址)
// ============================================
/*
 * 经典的 shared memory tree reduction。
 *
 * 问题: 早期版本有 warp divergence 和 bank conflict。
 * 本版本使用交错寻址 (reversed loop) 避免 bank conflict:
 *   stride = blockDim.x / 2, blockDim.x / 4, ..., 1
 *   每次将相邻的两个元素相加
 */
__global__ void reduce_naive(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    // 加载到 shared memory
    sdata[tid] = (global_id < n) ? input[global_id] : 0.0f;
    __syncthreads();

    // Tree reduction (交错寻址，无 bank conflict)
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // Block 的结果写到 output
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// ============================================
// Version 2: Grid-stride + Warp Shuffle Reduce
// ============================================
/*
 * 两个关键优化:
 * 1. Grid-stride loop: 每个线程处理多个元素，减少 block 数量
 * 2. Warp shuffle: 用 __shfl_down_sync 替代 shared memory reduce
 *    - 寄存器级别操作，零延迟
 *    - 不需要 __syncthreads()
 *    - 一个 warp (32 threads) 只需 5 步就能 reduce 完
 */
__device__ float warp_reduce_sum(float val) {
    // 5 步 reduce: 32 → 16 → 8 → 4 → 2 → 1
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void reduce_warp_shuffle(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_stride = blockDim.x * gridDim.x;

    // Step 1: Grid-stride loop，每个线程累加多个元素
    float thread_sum = 0.0f;
    for (int i = global_id; i < n; i += grid_stride) {
        thread_sum += input[i];
    }

    // Step 2: Warp-level reduce (32 threads → 1 value)
    thread_sum = warp_reduce_sum(thread_sum);

    // Step 3: Warp 间 reduce via shared memory
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;

    if (lane_id == 0) {
        sdata[warp_id] = thread_sum;
    }
    __syncthreads();

    // 第一个 warp 做最终 reduce
    if (warp_id == 0) {
        float val = (lane_id < num_warps) ? sdata[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane_id == 0) {
            output[blockIdx.x] = val;
        }
    }
}

// ============================================
// Version 3: Vectorized + Warp Shuffle (最优版本)
// ============================================
/*
 * 在 V2 基础上加入 float4 向量化加载:
 * - 每次加载 128 bits (4 个 float)
 * - 减少内存事务数量
 * - 更好地利用内存带宽
 */
__global__ void reduce_vectorized(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int grid_stride = blockDim.x * gridDim.x;

    // Step 1: 向量化 grid-stride loop
    float thread_sum = 0.0f;
    int vec_n = n / 4;
    const float4* input_vec = reinterpret_cast<const float4*>(input);

    int vec_global_id = blockIdx.x * blockDim.x + tid;
    int vec_grid_stride = grid_stride;

    for (int i = vec_global_id; i < vec_n; i += vec_grid_stride) {
        float4 val = input_vec[i];
        thread_sum += val.x + val.y + val.z + val.w;
    }

    // 处理剩余元素 (n 不是 4 的倍数时)
    int remaining_start = vec_n * 4;
    for (int i = remaining_start + vec_global_id; i < n; i += grid_stride) {
        thread_sum += input[i];
    }

    // Step 2: Warp reduce
    thread_sum = warp_reduce_sum(thread_sum);

    // Step 3: Block reduce
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;

    if (lane_id == 0) {
        sdata[warp_id] = thread_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < num_warps) ? sdata[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane_id == 0) {
            output[blockIdx.x] = val;
        }
    }
}

// ============================================
// Max Reduce (用于 Softmax)
// ============================================
__device__ float warp_reduce_max(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__global__ void reduce_max_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_stride = blockDim.x * gridDim.x;

    float thread_max = -INFINITY;
    for (int i = global_id; i < n; i += grid_stride) {
        thread_max = fmaxf(thread_max, input[i]);
    }

    thread_max = warp_reduce_max(thread_max);

    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;

    if (lane_id == 0) sdata[warp_id] = thread_max;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < num_warps) ? sdata[lane_id] : -INFINITY;
        val = warp_reduce_max(val);
        if (lane_id == 0) output[blockIdx.x] = val;
    }
}

// ============================================
// CPU 参考实现
// ============================================
float reduce_sum_cpu(const float* input, int n) {
    double sum = 0.0;  // 用 double 避免大数组精度损失
    for (int i = 0; i < n; i++) sum += input[i];
    return (float)sum;
}

float reduce_max_cpu(const float* input, int n) {
    float max_val = -INFINITY;
    for (int i = 0; i < n; i++) {
        if (input[i] > max_val) max_val = input[i];
    }
    return max_val;
}

// ============================================
// 辅助: 两阶段 reduce
// ============================================
float gpu_reduce_two_stage(
    void (*kernel)(const float*, float*, int),
    const float* d_input, int n,
    int block_size, int num_blocks
) {
    float *d_partial, *d_result;
    CUDA_CHECK(cudaMalloc(&d_partial, num_blocks * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(float)));

    int shared_mem = (block_size / WARP_SIZE) * sizeof(float);

    // Stage 1: 每个 block 输出一个部分和
    kernel<<<num_blocks, block_size, shared_mem>>>(d_input, d_partial, n);

    // Stage 2: 归约所有 block 的部分和
    kernel<<<1, block_size, shared_mem>>>(d_partial, d_result, num_blocks);

    float result;
    CUDA_CHECK(cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_partial));
    CUDA_CHECK(cudaFree(d_result));
    return result;
}

// ============================================
// Main
// ============================================
int main() {
    printf("============================================================\n");
    printf("  CUDA 算子: Parallel Reduce (Sum + Max)\n");
    printf("============================================================\n");

    srand(42);
    const int N = 1024 * 1024 * 16;  // 16M elements
    size_t size = N * sizeof(float);

    float *h_input = (float*)malloc(size);
    for (int i = 0; i < N; i++)
        h_input[i] = (float)(rand() % 100) / 100.0f;

    float ref_sum = reduce_sum_cpu(h_input, N);
    float ref_max = reduce_max_cpu(h_input, N);

    float *d_input;
    CUDA_CHECK(cudaMalloc(&d_input, size));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    int block_size = 256;
    int num_blocks = 256;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    int warmup = 10, repeat = 100;
    float ms;

    printf("\n  CPU 参考: sum = %.4f, max = %.4f\n", ref_sum, ref_max);
    printf("  数据量: %d 元素 (%.1f MB)\n", N, size / 1e6);

    // --- V1: Naive ---
    {
        int naive_blocks = (N + block_size - 1) / block_size;
        if (naive_blocks > 65535) naive_blocks = 65535;
        int shared_mem = block_size * sizeof(float);

        float *d_partial, *d_result;
        CUDA_CHECK(cudaMalloc(&d_partial, naive_blocks * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_result, sizeof(float)));

        for (int i = 0; i < warmup; i++) {
            reduce_naive<<<naive_blocks, block_size, shared_mem>>>(d_input, d_partial, N);
            reduce_naive<<<1, block_size, shared_mem>>>(d_partial, d_result, naive_blocks);
        }
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        for (int i = 0; i < repeat; i++) {
            reduce_naive<<<naive_blocks, block_size, shared_mem>>>(d_input, d_partial, N);
            reduce_naive<<<1, block_size, shared_mem>>>(d_partial, d_result, naive_blocks);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);

        float result;
        CUDA_CHECK(cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost));
        float rel_err = fabsf(result - ref_sum) / fabsf(ref_sum);
        float bandwidth = size / (ms / repeat / 1000.0f) / 1e9;

        printf("\n  V1 Naive (Shared Memory Tree):\n");
        printf("    时间: %.3f ms | 带宽: %.1f GB/s | 相对误差: %.2e %s\n",
               ms / repeat, bandwidth, rel_err, rel_err < 1e-3 ? "✅" : "❌");

        CUDA_CHECK(cudaFree(d_partial));
        CUDA_CHECK(cudaFree(d_result));
    }

    // --- V2: Warp Shuffle ---
    {
        float result = gpu_reduce_two_stage(reduce_warp_shuffle, d_input, N, block_size, num_blocks);
        float rel_err = fabsf(result - ref_sum) / fabsf(ref_sum);

        // Benchmark
        float *d_partial, *d_result;
        CUDA_CHECK(cudaMalloc(&d_partial, num_blocks * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_result, sizeof(float)));
        int shared_mem = (block_size / WARP_SIZE) * sizeof(float);

        for (int i = 0; i < warmup; i++) {
            reduce_warp_shuffle<<<num_blocks, block_size, shared_mem>>>(d_input, d_partial, N);
            reduce_warp_shuffle<<<1, block_size, shared_mem>>>(d_partial, d_result, num_blocks);
        }
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        for (int i = 0; i < repeat; i++) {
            reduce_warp_shuffle<<<num_blocks, block_size, shared_mem>>>(d_input, d_partial, N);
            reduce_warp_shuffle<<<1, block_size, shared_mem>>>(d_partial, d_result, num_blocks);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);

        float bandwidth = size / (ms / repeat / 1000.0f) / 1e9;
        printf("\n  V2 Warp Shuffle + Grid-stride:\n");
        printf("    时间: %.3f ms | 带宽: %.1f GB/s | 相对误差: %.2e %s\n",
               ms / repeat, bandwidth, rel_err, rel_err < 1e-3 ? "✅" : "❌");

        CUDA_CHECK(cudaFree(d_partial));
        CUDA_CHECK(cudaFree(d_result));
    }

    // --- V3: Vectorized ---
    {
        float result = gpu_reduce_two_stage(reduce_vectorized, d_input, N, block_size, num_blocks);
        float rel_err = fabsf(result - ref_sum) / fabsf(ref_sum);

        float *d_partial, *d_result;
        CUDA_CHECK(cudaMalloc(&d_partial, num_blocks * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_result, sizeof(float)));
        int shared_mem = (block_size / WARP_SIZE) * sizeof(float);

        for (int i = 0; i < warmup; i++) {
            reduce_vectorized<<<num_blocks, block_size, shared_mem>>>(d_input, d_partial, N);
            reduce_vectorized<<<1, block_size, shared_mem>>>(d_partial, d_result, num_blocks);
        }
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        for (int i = 0; i < repeat; i++) {
            reduce_vectorized<<<num_blocks, block_size, shared_mem>>>(d_input, d_partial, N);
            reduce_vectorized<<<1, block_size, shared_mem>>>(d_partial, d_result, num_blocks);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);

        float bandwidth = size / (ms / repeat / 1000.0f) / 1e9;
        printf("\n  V3 Vectorized (float4) + Warp Shuffle:\n");
        printf("    时间: %.3f ms | 带宽: %.1f GB/s | 相对误差: %.2e %s\n",
               ms / repeat, bandwidth, rel_err, rel_err < 1e-3 ? "✅" : "❌");

        CUDA_CHECK(cudaFree(d_partial));
        CUDA_CHECK(cudaFree(d_result));
    }

    // --- Max Reduce ---
    {
        float result = gpu_reduce_two_stage(reduce_max_kernel, d_input, N, block_size, num_blocks);
        float diff = fabsf(result - ref_max);
        printf("\n  Max Reduce (Warp Shuffle):\n");
        printf("    结果: %.4f | 误差: %.2e %s\n", result, diff, diff < 1e-5 ? "✅" : "❌");
    }

    printf("\n  💡 面试要点:\n");
    printf("    1. Naive reduce 的 warp divergence: if (tid %% stride == 0) 导致一半线程空闲\n");
    printf("    2. 交错寻址: stride 从大到小，避免 bank conflict\n");
    printf("    3. __shfl_down_sync: warp 内寄存器级别交换，比 shared memory 快\n");
    printf("    4. Grid-stride loop: 每个线程处理多个元素，减少 block 数量\n");
    printf("    5. 两阶段 reduce: block 内 reduce → block 间 reduce\n");
    printf("    6. float4 向量化: 减少内存事务，提升带宽利用率\n");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    CUDA_CHECK(cudaFree(d_input));
    free(h_input);

    return 0;
}
