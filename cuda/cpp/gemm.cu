/*
 * CUDA 算子: GEMM (General Matrix Multiplication)
 * ================================================
 * GEMM 是 LLM 中计算量最大的算子，面试考频最高。
 * 本文件包含 4 个版本，从 naive 到优化，逐步展示 CUDA 优化思路。
 *
 * 面试考点:
 * ---------
 * 1. Naive GEMM 的性能瓶颈: Global Memory 访问延迟
 * 2. Shared Memory Tiling: 减少 Global Memory 访问次数
 * 3. 计算访存比 (Arithmetic Intensity) 分析
 * 4. Bank Conflict 和 Padding 技巧
 * 5. 寄存器 Tiling (Thread-level tiling)
 * 6. 向量化访存 (float4 / LDS.128)
 * 7. cuBLAS 的性能为什么这么高 (Tensor Core, warp-level MMA)
 *
 * 编译运行:
 *   nvcc -O3 -arch=sm_89 gemm.cu -o gemm && ./gemm
 *   (sm_89 对应 RTX 5070, 如果是其他卡请修改)
 *
 * 原理:
 * -----
 * C[M,N] = A[M,K] @ B[K,N]
 *
 * Naive: 每个线程计算 C 的一个元素
 *   for k in range(K):
 *       C[row][col] += A[row][k] * B[k][col]
 *   问题: 每次循环都要从 Global Memory 读 A 和 B，延迟巨大
 *
 * Tiling: 将 A 和 B 分块加载到 Shared Memory
 *   for tile in range(K / TILE_SIZE):
 *       加载 A_tile 和 B_tile 到 Shared Memory  (一次加载，多次使用)
 *       __syncthreads()
 *       for k in range(TILE_SIZE):
 *           C[row][col] += A_shared[ty][k] * B_shared[k][tx]
 *       __syncthreads()
 *
 *   计算访存比提升: 从 O(1) 提升到 O(TILE_SIZE)
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// ============================================
// 错误检查宏
// ============================================
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
// Version 1: Naive GEMM
// ============================================
/*
 * 每个线程计算 C 的一个元素。
 * 
 * 性能瓶颈分析:
 * - 每个线程执行 K 次乘加，每次需要从 Global Memory 读 A[row][k] 和 B[k][col]
 * - Global Memory 延迟: ~400-800 cycles
 * - 计算访存比: 2 FLOPs / (2 * 4 bytes) = 0.25 FLOPs/byte (极低)
 * - GPU 的峰值计算访存比通常 > 100 FLOPs/byte，严重 memory-bound
 */
__global__ void gemm_naive(
    const float* __restrict__ A,  // [M, K]
    const float* __restrict__ B,  // [K, N]
    float* __restrict__ C,        // [M, N]
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// ============================================
// Version 2: Shared Memory Tiling
// ============================================
/*
 * 核心优化: 将 A 和 B 的子块加载到 Shared Memory，复用数据。
 *
 * Shared Memory 特性:
 * - 延迟: ~5 cycles (vs Global Memory ~400 cycles)
 * - 带宽: ~19 TB/s (vs HBM ~1-2 TB/s)
 * - 大小: 通常 48-164 KB per SM
 *
 * Tiling 后的计算访存比:
 * - 每个 tile 加载: TILE_SIZE * TILE_SIZE * 2 个 float (A_tile + B_tile)
 * - 每个 tile 计算: TILE_SIZE * TILE_SIZE * TILE_SIZE * 2 FLOPs
 * - 计算访存比: TILE_SIZE / 2 (TILE_SIZE=32 时为 16x 提升!)
 */
#define TILE_SIZE 32

__global__ void gemm_shared_memory(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    // Shared Memory: 每个 block 共享的高速缓存
    __shared__ float A_shared[TILE_SIZE][TILE_SIZE];
    __shared__ float B_shared[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float sum = 0.0f;

    // 沿 K 维度分块遍历
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int tile = 0; tile < num_tiles; tile++) {
        // 协作加载: block 内所有线程一起将 A_tile 和 B_tile 加载到 Shared Memory
        int a_col = tile * TILE_SIZE + tx;
        int b_row = tile * TILE_SIZE + ty;

        // 边界检查
        A_shared[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        B_shared[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        // 同步: 确保所有线程都完成了加载
        __syncthreads();

        // 计算: 使用 Shared Memory 中的数据 (快 ~80x)
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += A_shared[ty][k] * B_shared[k][tx];
        }

        // 同步: 确保所有线程都完成了计算，再加载下一个 tile
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// ============================================
// Version 3: Shared Memory + Bank Conflict 优化
// ============================================
/*
 * Bank Conflict 问题:
 * - Shared Memory 被分为 32 个 bank (每个 bank 4 bytes 宽)
 * - 同一 warp 的线程访问同一 bank 的不同地址时，会串行化 (bank conflict)
 * - B_shared[k][tx]: 当 k 固定时，tx=0,1,...,31 访问连续 bank → 无冲突 ✓
 * - A_shared[ty][k]: 当 ty 固定时，k=0,1,...,31 访问连续 bank → 无冲突 ✓
 *
 * 但如果矩阵维度恰好是 32 的倍数，列访问可能产生冲突。
 * 解决方案: Padding，将 shared memory 的列数 +1
 */
#define TILE_SIZE_PADDED 33  // 32 + 1 padding to avoid bank conflict

__global__ void gemm_shared_memory_no_bank_conflict(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    // Padding: 列数 +1，错开 bank 访问
    __shared__ float A_shared[TILE_SIZE][TILE_SIZE_PADDED];
    __shared__ float B_shared[TILE_SIZE][TILE_SIZE_PADDED];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float sum = 0.0f;

    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int tile = 0; tile < num_tiles; tile++) {
        int a_col = tile * TILE_SIZE + tx;
        int b_row = tile * TILE_SIZE + ty;

        A_shared[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        B_shared[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();

        // 使用 #pragma unroll 提示编译器展开循环，减少循环开销
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += A_shared[ty][k] * B_shared[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// ============================================
// Version 4: 寄存器 Tiling (Thread-level Tiling)
// ============================================
/*
 * 进一步优化: 每个线程计算 C 的一个 TM x TN 的小块，而不是单个元素。
 *
 * 优势:
 * - 每个线程从 Shared Memory 加载的数据被复用 TM 或 TN 次
 * - 计算访存比进一步提升: 从 O(TILE_SIZE) 到 O(TILE_SIZE * TM * TN / (TM + TN))
 * - 减少线程数量，降低调度开销
 *
 * 这是 cuBLAS 等高性能库的核心思想之一。
 */
#define BM 64       // Block 处理的 M 维度大小
#define BN 64       // Block 处理的 N 维度大小
#define BK 16       // 沿 K 维度的 tile 大小
#define TM 4        // 每个线程计算的 M 维度大小
#define TN 4        // 每个线程计算的 N 维度大小

__global__ void gemm_register_tiling(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    // Block 内的线程排布: (BN/TN) x (BM/TM) = 16 x 16 = 256 threads
    const int thread_col = threadIdx.x % (BN / TN);  // 0..15
    const int thread_row = threadIdx.x / (BN / TN);  // 0..15

    // 当前 block 处理的 C 子矩阵的起始位置
    const int block_row_start = blockIdx.y * BM;
    const int block_col_start = blockIdx.x * BN;

    // Shared Memory
    __shared__ float A_shared[BM][BK];
    __shared__ float B_shared[BK][BN];

    // 寄存器: 每个线程的局部累加器 (TM x TN 个元素)
    float thread_results[TM][TN] = {0.0f};
    // 寄存器: 缓存从 Shared Memory 加载的 A 和 B 的列/行
    float reg_a[TM];
    float reg_b[TN];

    // 计算加载 Shared Memory 时每个线程负责的位置
    const int num_threads = (BM / TM) * (BN / TN);  // 256
    const int a_loads_per_thread = (BM * BK) / num_threads;  // 每个线程加载几个 A 元素
    const int b_loads_per_thread = (BK * BN) / num_threads;  // 每个线程加载几个 B 元素

    // 沿 K 维度分块
    for (int bk = 0; bk < K; bk += BK) {
        // 协作加载 A_shared[BM][BK] 和 B_shared[BK][BN]
        for (int load_idx = 0; load_idx < a_loads_per_thread; load_idx++) {
            int flat_idx = threadIdx.x * a_loads_per_thread + load_idx;
            int load_row = flat_idx / BK;
            int load_col = flat_idx % BK;
            int global_row = block_row_start + load_row;
            int global_col = bk + load_col;
            A_shared[load_row][load_col] =
                (global_row < M && global_col < K) ? A[global_row * K + global_col] : 0.0f;
        }

        for (int load_idx = 0; load_idx < b_loads_per_thread; load_idx++) {
            int flat_idx = threadIdx.x * b_loads_per_thread + load_idx;
            int load_row = flat_idx / BN;
            int load_col = flat_idx % BN;
            int global_row = bk + load_row;
            int global_col = block_col_start + load_col;
            B_shared[load_row][load_col] =
                (global_row < K && global_col < N) ? B[global_row * N + global_col] : 0.0f;
        }

        __syncthreads();

        // 计算: 每个线程计算 TM x TN 的子块
        for (int k = 0; k < BK; k++) {
            // 从 Shared Memory 加载到寄存器
            #pragma unroll
            for (int tm = 0; tm < TM; tm++) {
                reg_a[tm] = A_shared[thread_row * TM + tm][k];
            }
            #pragma unroll
            for (int tn = 0; tn < TN; tn++) {
                reg_b[tn] = B_shared[k][thread_col * TN + tn];
            }

            // 外积累加: TM x TN 次乘加
            #pragma unroll
            for (int tm = 0; tm < TM; tm++) {
                #pragma unroll
                for (int tn = 0; tn < TN; tn++) {
                    thread_results[tm][tn] += reg_a[tm] * reg_b[tn];
                }
            }
        }

        __syncthreads();
    }

    // 写回结果
    for (int tm = 0; tm < TM; tm++) {
        for (int tn = 0; tn < TN; tn++) {
            int global_row = block_row_start + thread_row * TM + tm;
            int global_col = block_col_start + thread_col * TN + tn;
            if (global_row < M && global_col < N) {
                C[global_row * N + global_col] = thread_results[tm][tn];
            }
        }
    }
}

// ============================================
// Version 5: Double Buffering (预取隐藏延迟)
// ============================================
/*
 * 核心思想: 在计算当前 tile 的同时，预取下一个 tile 到另一块 shared memory。
 *
 * 问题: V4 中每次 tile 循环都有:
 *   1. 加载 tile → __syncthreads() → 2. 计算 → __syncthreads() → 回到 1
 *   加载和计算是串行的，GPU 在加载时计算单元空闲。
 *
 * Double Buffering:
 *   使用两块 shared memory (buffer 0 和 buffer 1)
 *   - 计算 buffer[0] 的同时，加载数据到 buffer[1]
 *   - 计算 buffer[1] 的同时，加载数据到 buffer[0]
 *   - 加载和计算重叠执行 (overlap)
 *
 * 效果: 隐藏 Global Memory 加载延迟，提升 ~10-15% 性能。
 */
#define DB_BM 64
#define DB_BN 64
#define DB_BK 8
#define DB_TM 4
#define DB_TN 4

__global__ void gemm_double_buffering(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    const int thread_col = threadIdx.x % (DB_BN / DB_TN);
    const int thread_row = threadIdx.x / (DB_BN / DB_TN);

    const int block_row_start = blockIdx.y * DB_BM;
    const int block_col_start = blockIdx.x * DB_BN;

    // Double buffer: 两块 shared memory
    __shared__ float A_shared[2][DB_BM][DB_BK];
    __shared__ float B_shared[2][DB_BK][DB_BN];

    float thread_results[DB_TM][DB_TN] = {0.0f};
    float reg_a[DB_TM];
    float reg_b[DB_TN];

    const int num_threads = (DB_BM / DB_TM) * (DB_BN / DB_TN);
    const int a_loads = (DB_BM * DB_BK) / num_threads;
    const int b_loads = (DB_BK * DB_BN) / num_threads;

    // 预加载第一个 tile 到 buffer 0
    int buf = 0;
    for (int ld = 0; ld < a_loads; ld++) {
        int flat = threadIdx.x * a_loads + ld;
        int lr = flat / DB_BK, lc = flat % DB_BK;
        int gr = block_row_start + lr, gc = lc;
        A_shared[buf][lr][lc] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
    }
    for (int ld = 0; ld < b_loads; ld++) {
        int flat = threadIdx.x * b_loads + ld;
        int lr = flat / DB_BN, lc = flat % DB_BN;
        int gr = lr, gc = block_col_start + lc;
        B_shared[buf][lr][lc] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
    }
    __syncthreads();

    int num_tiles = (K + DB_BK - 1) / DB_BK;
    for (int tile = 0; tile < num_tiles; tile++) {
        int next_buf = 1 - buf;
        int next_bk = (tile + 1) * DB_BK;

        // 预取下一个 tile 到 next_buf (与当前计算重叠)
        if (tile + 1 < num_tiles) {
            for (int ld = 0; ld < a_loads; ld++) {
                int flat = threadIdx.x * a_loads + ld;
                int lr = flat / DB_BK, lc = flat % DB_BK;
                int gr = block_row_start + lr, gc = next_bk + lc;
                A_shared[next_buf][lr][lc] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
            }
            for (int ld = 0; ld < b_loads; ld++) {
                int flat = threadIdx.x * b_loads + ld;
                int lr = flat / DB_BN, lc = flat % DB_BN;
                int gr = next_bk + lr, gc = block_col_start + lc;
                B_shared[next_buf][lr][lc] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
            }
        }

        // 计算当前 buffer
        for (int k = 0; k < DB_BK; k++) {
            #pragma unroll
            for (int tm = 0; tm < DB_TM; tm++)
                reg_a[tm] = A_shared[buf][thread_row * DB_TM + tm][k];
            #pragma unroll
            for (int tn = 0; tn < DB_TN; tn++)
                reg_b[tn] = B_shared[buf][k][thread_col * DB_TN + tn];
            #pragma unroll
            for (int tm = 0; tm < DB_TM; tm++)
                #pragma unroll
                for (int tn = 0; tn < DB_TN; tn++)
                    thread_results[tm][tn] += reg_a[tm] * reg_b[tn];
        }

        __syncthreads();
        buf = next_buf;
    }

    for (int tm = 0; tm < DB_TM; tm++)
        for (int tn = 0; tn < DB_TN; tn++) {
            int gr = block_row_start + thread_row * DB_TM + tm;
            int gc = block_col_start + thread_col * DB_TN + tn;
            if (gr < M && gc < N)
                C[gr * N + gc] = thread_results[tm][tn];
        }
}

// ============================================
// Version 6: Warptiling (Warp 级别协作)
// ============================================
/*
 * 核心思想: 在 thread tiling 和 block tiling 之间加入 warp tiling 层次。
 *
 * 层次结构:
 *   Block Tile (BM x BN)
 *     → Warp Tile (WM x WN): 一个 warp 负责的子矩阵
 *       → Thread Tile (TM x TN): 一个线程负责的子矩阵
 *
 * 优势:
 * - Warp 内线程协作访问 shared memory，更好的访存模式
 * - 减少 warp 间的 shared memory bank conflict
 * - 更接近 cuBLAS 的实现思路
 *
 * 这是 siboehm/SGEMM_CUDA 中达到 cuBLAS 93.7% 性能的关键优化。
 */
#define WT_BM 128
#define WT_BN 128
#define WT_BK 16
#define WT_WM 64      // Warp tile M
#define WT_WN 32      // Warp tile N
#define WT_TM 8       // Thread tile M
#define WT_TN 4       // Thread tile N
#define WT_NUM_THREADS 256

__global__ void gemm_warptiling(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    const int block_row = blockIdx.y * WT_BM;
    const int block_col = blockIdx.x * WT_BN;

    // Warp 在 block 内的位置
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    // Block 内 warp 的 2D 排布
    const int warps_per_row = WT_BN / WT_WN;  // 128/32 = 4
    const int warp_row = warp_id / warps_per_row;
    const int warp_col = warp_id % warps_per_row;

    // Thread 在 warp tile 内的位置
    const int threads_per_warp_row = WT_WN / WT_TN;  // 32/4 = 8
    const int thread_row_in_warp = lane_id / threads_per_warp_row;
    const int thread_col_in_warp = lane_id % threads_per_warp_row;

    __shared__ float A_shared[WT_BM][WT_BK];
    __shared__ float B_shared[WT_BK][WT_BN];

    float thread_results[WT_TM][WT_TN] = {0.0f};
    float reg_a[WT_TM];
    float reg_b[WT_TN];

    const int a_loads = (WT_BM * WT_BK) / WT_NUM_THREADS;
    const int b_loads = (WT_BK * WT_BN) / WT_NUM_THREADS;

    for (int bk = 0; bk < K; bk += WT_BK) {
        // 协作加载
        for (int ld = 0; ld < a_loads; ld++) {
            int flat = threadIdx.x * a_loads + ld;
            int lr = flat / WT_BK, lc = flat % WT_BK;
            int gr = block_row + lr, gc = bk + lc;
            A_shared[lr][lc] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
        }
        for (int ld = 0; ld < b_loads; ld++) {
            int flat = threadIdx.x * b_loads + ld;
            int lr = flat / WT_BN, lc = flat % WT_BN;
            int gr = bk + lr, gc = block_col + lc;
            B_shared[lr][lc] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
        }
        __syncthreads();

        // 计算: 每个线程在其 warp tile 内计算 TM x TN 子块
        for (int k = 0; k < WT_BK; k++) {
            // 从 shared memory 加载到寄存器
            // 注意: 使用 warp tile 的偏移
            int a_row_base = warp_row * WT_WM + thread_row_in_warp * WT_TM;
            int b_col_base = warp_col * WT_WN + thread_col_in_warp * WT_TN;

            #pragma unroll
            for (int tm = 0; tm < WT_TM; tm++)
                reg_a[tm] = A_shared[a_row_base + tm][k];
            #pragma unroll
            for (int tn = 0; tn < WT_TN; tn++)
                reg_b[tn] = B_shared[k][b_col_base + tn];

            #pragma unroll
            for (int tm = 0; tm < WT_TM; tm++)
                #pragma unroll
                for (int tn = 0; tn < WT_TN; tn++)
                    thread_results[tm][tn] += reg_a[tm] * reg_b[tn];
        }
        __syncthreads();
    }

    // 写回
    int out_row_base = block_row + warp_row * WT_WM + thread_row_in_warp * WT_TM;
    int out_col_base = block_col + warp_col * WT_WN + thread_col_in_warp * WT_TN;

    for (int tm = 0; tm < WT_TM; tm++)
        for (int tn = 0; tn < WT_TN; tn++) {
            int gr = out_row_base + tm;
            int gc = out_col_base + tn;
            if (gr < M && gc < N)
                C[gr * N + gc] = thread_results[tm][tn];
        }
}

// ============================================
// CPU 参考实现
// ============================================
void gemm_cpu(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// ============================================
// 工具函数
// ============================================
void init_matrix(float* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)(rand() % 100) / 100.0f;
    }
}

float max_diff(const float* a, const float* b, int n) {
    float max_d = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > max_d) max_d = d;
    }
    return max_d;
}

float benchmark_kernel(
    void (*kernel)(const float*, const float*, float*, int, int, int),
    const float* d_A, const float* d_B, float* d_C,
    int M, int N, int K,
    dim3 grid, dim3 block,
    int warmup, int repeat
) {
    // Warmup
    for (int i = 0; i < warmup; i++) {
        kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    cudaDeviceSynchronize();

    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < repeat; i++) {
        kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms / repeat;
}

// ============================================
// Main: 正确性验证 + 性能对比
// ============================================
int main() {
    printf("============================================================\n");
    printf("  CUDA 算子: GEMM (6 个版本性能对比)\n");
    printf("============================================================\n");

    srand(42);

    const int M = 1024, N = 1024, K = 1024;
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    // Host 内存
    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_C_ref = (float*)malloc(size_C);
    float *h_C_gpu = (float*)malloc(size_C);

    init_matrix(h_A, M, K);
    init_matrix(h_B, K, N);

    // CPU 参考结果
    printf("\n  计算 CPU 参考结果...\n");
    gemm_cpu(h_A, h_B, h_C_ref, M, N, K);

    // Device 内存
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));

    float gflops = 2.0f * M * N * K / 1e9;
    int warmup = 5, repeat = 20;

    // --- Version 1: Naive ---
    {
        dim3 block(32, 32);
        dim3 grid((N + 31) / 32, (M + 31) / 32);
        float ms = benchmark_kernel(gemm_naive, d_A, d_B, d_C, M, N, K, grid, block, warmup, repeat);
        CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, size_C, cudaMemcpyDeviceToHost));
        float diff = max_diff(h_C_ref, h_C_gpu, M * N);
        printf("\n  V1 Naive:\n");
        printf("    时间: %.3f ms | GFLOPS: %.1f | 误差: %.2e %s\n",
               ms, gflops / ms * 1000, diff, diff < 1e-2 ? "✅" : "❌");
    }

    // --- Version 2: Shared Memory Tiling ---
    {
        dim3 block(TILE_SIZE, TILE_SIZE);
        dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
        float ms = benchmark_kernel(gemm_shared_memory, d_A, d_B, d_C, M, N, K, grid, block, warmup, repeat);
        CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, size_C, cudaMemcpyDeviceToHost));
        float diff = max_diff(h_C_ref, h_C_gpu, M * N);
        printf("\n  V2 Shared Memory Tiling:\n");
        printf("    时间: %.3f ms | GFLOPS: %.1f | 误差: %.2e %s\n",
               ms, gflops / ms * 1000, diff, diff < 1e-2 ? "✅" : "❌");
    }

    // --- Version 3: Shared Memory + No Bank Conflict ---
    {
        dim3 block(TILE_SIZE, TILE_SIZE);
        dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
        float ms = benchmark_kernel(gemm_shared_memory_no_bank_conflict, d_A, d_B, d_C, M, N, K, grid, block, warmup, repeat);
        CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, size_C, cudaMemcpyDeviceToHost));
        float diff = max_diff(h_C_ref, h_C_gpu, M * N);
        printf("\n  V3 Shared Memory + Padding (No Bank Conflict):\n");
        printf("    时间: %.3f ms | GFLOPS: %.1f | 误差: %.2e %s\n",
               ms, gflops / ms * 1000, diff, diff < 1e-2 ? "✅" : "❌");
    }

    // --- Version 4: Register Tiling ---
    {
        dim3 block((BN / TN) * (BM / TM));  // 256 threads
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
        float ms = benchmark_kernel(gemm_register_tiling, d_A, d_B, d_C, M, N, K, grid, block, warmup, repeat);
        CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, size_C, cudaMemcpyDeviceToHost));
        float diff = max_diff(h_C_ref, h_C_gpu, M * N);
        printf("\n  V4 Register Tiling (TM=%d, TN=%d):\n", TM, TN);
        printf("    时间: %.3f ms | GFLOPS: %.1f | 误差: %.2e %s\n",
               ms, gflops / ms * 1000, diff, diff < 1e-2 ? "✅" : "❌");
    }

    // --- Version 5: Double Buffering ---
    {
        dim3 block((DB_BN / DB_TN) * (DB_BM / DB_TM));
        dim3 grid((N + DB_BN - 1) / DB_BN, (M + DB_BM - 1) / DB_BM);
        float ms = benchmark_kernel(gemm_double_buffering, d_A, d_B, d_C, M, N, K, grid, block, warmup, repeat);
        CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, size_C, cudaMemcpyDeviceToHost));
        float diff = max_diff(h_C_ref, h_C_gpu, M * N);
        printf("\n  V5 Double Buffering:\n");
        printf("    时间: %.3f ms | GFLOPS: %.1f | 误差: %.2e %s\n",
               ms, gflops / ms * 1000, diff, diff < 1e-2 ? "✅" : "❌");
    }

    // --- Version 6: Warptiling ---
    {
        dim3 block(WT_NUM_THREADS);
        dim3 grid((N + WT_BN - 1) / WT_BN, (M + WT_BM - 1) / WT_BM);
        float ms = benchmark_kernel(gemm_warptiling, d_A, d_B, d_C, M, N, K, grid, block, warmup, repeat);
        CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, size_C, cudaMemcpyDeviceToHost));
        float diff = max_diff(h_C_ref, h_C_gpu, M * N);
        printf("\n  V6 Warptiling (WM=%d, WN=%d, TM=%d, TN=%d):\n", WT_WM, WT_WN, WT_TM, WT_TN);
        printf("    时间: %.3f ms | GFLOPS: %.1f | 误差: %.2e %s\n",
               ms, gflops / ms * 1000, diff, diff < 1e-2 ? "✅" : "❌");
    }

    printf("\n  GFLOPS 计算: 2 * M * N * K = 2 * %d * %d * %d = %.2f GFLOPS\n", M, N, K, gflops);
    printf("\n  优化路线:\n");
    printf("    V1 Naive → V2 Shared Memory Tiling → V3 Bank Conflict Padding\n");
    printf("    → V4 Register Tiling → V5 Double Buffering → V6 Warptiling\n");
    printf("\n  💡 面试要点:\n");
    printf("    1. Naive → Shared Memory: 减少 Global Memory 访问，计算访存比提升 TILE_SIZE 倍\n");
    printf("    2. Bank Conflict: Padding 列数 +1 错开 bank 访问\n");
    printf("    3. Register Tiling: 每个线程计算 TMxTN 子块，进一步提升数据复用\n");
    printf("    4. Double Buffering: 预取下一个 tile，隐藏 Global Memory 加载延迟\n");
    printf("    5. Warptiling: Block→Warp→Thread 三级 tiling，减少 warp 间 bank conflict\n");
    printf("    6. cuBLAS 还用了: Tensor Core (WMMA/MMA), 更精细的 warp 调度\n");
    printf("    7. 实际 LLM 中 GEMM 占 >60%% 计算量 (QKV projection, FFN)\n");

    // 清理
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C_ref);
    free(h_C_gpu);

    return 0;
}
