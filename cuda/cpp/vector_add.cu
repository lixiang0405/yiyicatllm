/*
 * CUDA 算子: Vector Add (向量加法)
 * =================================
 * CUDA 编程的 Hello World，理解 GPU 并行计算的基本概念。
 *
 * 面试考点:
 * ---------
 * 1. GPU 线程层次: Grid → Block → Thread
 * 2. 线程索引: global_id = blockIdx.x * blockDim.x + threadIdx.x
 * 3. Warp: 32 个线程为一组，SIMT 执行
 * 4. 合并访存 (Coalesced Access): 相邻线程访问相邻内存地址
 * 5. BLOCK_SIZE 选择: 32 的倍数 (对齐 warp)
 *
 * 编译运行:
 *   nvcc -O3 -arch=sm_89 vector_add.cu -o vector_add && ./vector_add
 *
 * GPU 线程层次:
 *   Grid (网格)
 *   ├── Block 0 (线程块)
 *   │   ├── Warp 0: Thread 0-31
 *   │   ├── Warp 1: Thread 32-63
 *   │   └── ...
 *   ├── Block 1
 *   │   ├── Warp 0: Thread 0-31
 *   │   └── ...
 *   └── ...
 */

#include <stdio.h>
#include <stdlib.h>
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

// ============================================
// Version 1: 基础版
// ============================================
/*
 * 每个线程处理一个元素。
 * 
 * 内存访问模式:
 *   Thread 0 访问 a[0], b[0] → 写 c[0]
 *   Thread 1 访问 a[1], b[1] → 写 c[1]
 *   ...
 *   相邻线程访问相邻地址 → 合并访存 (Coalesced Access) ✓
 */
__global__ void vector_add_v1(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// ============================================
// Version 2: 向量化访存 (float4)
// ============================================
/*
 * 每个线程处理 4 个元素 (128 bits 一次加载)。
 * 
 * 优势:
 * - 减少内存事务数量 (1/4)
 * - 更好地利用内存带宽
 * - 对于简单的 elementwise 操作效果显著
 */
__global__ void vector_add_v2_vectorized(
    const float4* __restrict__ a,
    const float4* __restrict__ b,
    float4* __restrict__ c,
    int n_vec
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_vec) {
        float4 va = a[idx];
        float4 vb = b[idx];
        float4 vc;
        vc.x = va.x + vb.x;
        vc.y = va.y + vb.y;
        vc.z = va.z + vb.z;
        vc.w = va.w + vb.w;
        c[idx] = vc;
    }
}

// ============================================
// Version 3: Grid-stride Loop
// ============================================
/*
 * Grid-stride loop: 每个线程处理多个元素，步长为 grid 总线程数。
 * 
 * 优势:
 * - 一个 kernel 可以处理任意大小的数据
 * - 不需要根据数据大小调整 grid 大小
 * - 更好的 GPU 占用率 (occupancy)
 *
 * 这是 CUDA 编程的最佳实践之一。
 */
__global__ void vector_add_v3_grid_stride(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride) {
        c[i] = a[i] + b[i];
    }
}

// ============================================
// Main
// ============================================
int main() {
    printf("============================================================\n");
    printf("  CUDA 算子: Vector Add (3 个版本)\n");
    printf("============================================================\n");

    srand(42);
    const int N = 1024 * 1024 * 16;  // 16M elements
    size_t size = N * sizeof(float);

    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c_ref = (float*)malloc(size);
    float *h_c_gpu = (float*)malloc(size);

    for (int i = 0; i < N; i++) {
        h_a[i] = (float)(rand() % 100) / 10.0f;
        h_b[i] = (float)(rand() % 100) / 10.0f;
        h_c_ref[i] = h_a[i] + h_b[i];
    }

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, size));
    CUDA_CHECK(cudaMalloc(&d_b, size));
    CUDA_CHECK(cudaMalloc(&d_c, size));
    CUDA_CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    int warmup = 10, repeat = 100;
    float ms, diff;
    int block_size = 256;

    // --- V1: 基础版 ---
    {
        int grid_size = (N + block_size - 1) / block_size;
        for (int i = 0; i < warmup; i++)
            vector_add_v1<<<grid_size, block_size>>>(d_a, d_b, d_c, N);
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        for (int i = 0; i < repeat; i++)
            vector_add_v1<<<grid_size, block_size>>>(d_a, d_b, d_c, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);

        CUDA_CHECK(cudaMemcpy(h_c_gpu, d_c, size, cudaMemcpyDeviceToHost));
        diff = 0;
        for (int i = 0; i < N; i++)
            diff = fmaxf(diff, fabsf(h_c_ref[i] - h_c_gpu[i]));

        float bandwidth = 3.0f * size / (ms / repeat / 1000.0f) / 1e9;
        printf("\n  V1 基础版:\n");
        printf("    时间: %.3f ms | 带宽: %.1f GB/s | 误差: %.2e %s\n",
               ms / repeat, bandwidth, diff, diff < 1e-5 ? "✅" : "❌");
    }

    // --- V2: 向量化版 ---
    {
        int n_vec = N / 4;
        int grid_size = (n_vec + block_size - 1) / block_size;
        for (int i = 0; i < warmup; i++)
            vector_add_v2_vectorized<<<grid_size, block_size>>>(
                (float4*)d_a, (float4*)d_b, (float4*)d_c, n_vec);
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        for (int i = 0; i < repeat; i++)
            vector_add_v2_vectorized<<<grid_size, block_size>>>(
                (float4*)d_a, (float4*)d_b, (float4*)d_c, n_vec);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);

        CUDA_CHECK(cudaMemcpy(h_c_gpu, d_c, size, cudaMemcpyDeviceToHost));
        diff = 0;
        for (int i = 0; i < N; i++)
            diff = fmaxf(diff, fabsf(h_c_ref[i] - h_c_gpu[i]));

        float bandwidth = 3.0f * size / (ms / repeat / 1000.0f) / 1e9;
        printf("\n  V2 向量化版 (float4):\n");
        printf("    时间: %.3f ms | 带宽: %.1f GB/s | 误差: %.2e %s\n",
               ms / repeat, bandwidth, diff, diff < 1e-5 ? "✅" : "❌");
    }

    // --- V3: Grid-stride Loop ---
    {
        int grid_size = 256;  // 固定 grid 大小
        for (int i = 0; i < warmup; i++)
            vector_add_v3_grid_stride<<<grid_size, block_size>>>(d_a, d_b, d_c, N);
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        for (int i = 0; i < repeat; i++)
            vector_add_v3_grid_stride<<<grid_size, block_size>>>(d_a, d_b, d_c, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);

        CUDA_CHECK(cudaMemcpy(h_c_gpu, d_c, size, cudaMemcpyDeviceToHost));
        diff = 0;
        for (int i = 0; i < N; i++)
            diff = fmaxf(diff, fabsf(h_c_ref[i] - h_c_gpu[i]));

        float bandwidth = 3.0f * size / (ms / repeat / 1000.0f) / 1e9;
        printf("\n  V3 Grid-stride Loop:\n");
        printf("    时间: %.3f ms | 带宽: %.1f GB/s | 误差: %.2e %s\n",
               ms / repeat, bandwidth, diff, diff < 1e-5 ? "✅" : "❌");
    }

    printf("\n  💡 面试要点:\n");
    printf("    1. 线程索引: idx = blockIdx.x * blockDim.x + threadIdx.x\n");
    printf("    2. 合并访存: 相邻线程访问相邻地址，一次内存事务服务整个 warp\n");
    printf("    3. float4 向量化: 减少内存事务数量，提升带宽利用率\n");
    printf("    4. Grid-stride loop: 固定 grid 大小，处理任意数据量\n");
    printf("    5. 带宽计算: 3 * N * sizeof(float) / time (读 a,b 写 c)\n");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    free(h_a);
    free(h_b);
    free(h_c_ref);
    free(h_c_gpu);

    return 0;
}
