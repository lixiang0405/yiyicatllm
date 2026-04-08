# 🔥 CUDA 算子实现 - LLM 面试核心知识

本目录包含 LLM 推理/训练中最重要的 CUDA 算子实现，提供 **Triton (Python)** 和 **CUDA C++ (.cu)** 两种版本：

- 📖 详细的原理注释和面试考点
- 💻 完整的 kernel 实现（多版本从 naive 到优化）
- ✅ 参考实现 + 正确性验证
- 📊 性能 benchmark 对比

## 算子清单

### Triton (Python) 版本

| 文件 | 算子 | 面试重要度 | 说明 |
|------|------|-----------|------|
| `01_vector_add.py` | Vector Add | ⭐⭐ | CUDA 入门，理解 Grid/Block/Thread 层次 |
| `02_softmax.py` | Softmax + Online Softmax | ⭐⭐⭐⭐⭐ | FlashAttention 的数学基础 |
| `03_rmsnorm.py` | RMSNorm | ⭐⭐⭐⭐ | Qwen/LLaMA 使用的归一化层 |
| `04_rope.py` | RoPE 旋转位置编码 | ⭐⭐⭐⭐ | 所有现代 LLM 的位置编码方式 |
| `05_flash_attention.py` | FlashAttention | ⭐⭐⭐⭐⭐ | 面试最高频，tiling + online softmax |
| `06_fused_add_rmsnorm.py` | Fused Add + RMSNorm | ⭐⭐⭐⭐ | 算子融合，memory-bound 优化核心 |

### CUDA C++ 版本 (`cpp/` 目录)

| 文件 | 算子 | 面试重要度 | 优化版本数 | 说明 |
|------|------|-----------|-----------|------|
| `cpp/gemm.cu` | **GEMM 矩阵乘法** | ⭐⭐⭐⭐⭐ | 4 个版本 | Naive → Shared Memory Tiling → Bank Conflict 优化 → Register Tiling |
| `cpp/softmax.cu` | Softmax | ⭐⭐⭐⭐⭐ | 2 个版本 | Safe Softmax + Online Softmax，含 warp reduction |
| `cpp/rmsnorm.cu` | RMSNorm | ⭐⭐⭐⭐ | 2 个版本 | 基础版 + float4 向量化访存 |
| `cpp/vector_add.cu` | Vector Add | ⭐⭐ | 3 个版本 | 基础 → float4 向量化 → Grid-stride Loop |
| `cpp/fused_add_rmsnorm.cu` | Fused Add+RMSNorm | ⭐⭐⭐⭐ | 2 个版本 | 非融合 vs 融合，含 IO 量分析 |

## 运行方式

### Triton (Python)

```bash
pip install triton

python cuda/01_vector_add.py
python cuda/02_softmax.py
python cuda/03_rmsnorm.py
python cuda/04_rope.py
python cuda/05_flash_attention.py
python cuda/06_fused_add_rmsnorm.py
```

### CUDA C++

```bash
cd cuda/cpp

# 编译所有算子
make all

# 编译并运行所有算子
make run

# 编译单个算子
make gemm
./gemm

# 清理
make clean
```

> **注意**: 修改 `Makefile` 中的 `ARCH` 以匹配你的 GPU：
> - RTX 5070: `sm_100` (Blackwell)
> - RTX 4090: `sm_89` (Ada Lovelace)
> - RTX 3090: `sm_86` (Ampere)
> - A100: `sm_80` (Ampere)

### GEMM 优化路线图 (面试最高频)

```
V1 Naive                    → 每个线程算 C 的一个元素，每次从 Global Memory 读 A/B
   ↓ 计算访存比: ~0.25 FLOPs/byte
V2 Shared Memory Tiling     → 分块加载到 Shared Memory，数据复用 TILE_SIZE 次
   ↓ 计算访存比: ~TILE_SIZE/2
V3 + Bank Conflict Padding  → 列数 +1 错开 bank 访问
   ↓
V4 Register Tiling          → 每个线程算 TM×TN 子块，外积累加
   ↓ 计算访存比: ~TILE_SIZE * TM * TN / (TM + TN)
cuBLAS (参考)               → Tensor Core + double buffering + warp MMA
```

## 与项目的关系

这些算子直接对应你的项目中 Qwen2.5-7B 模型的核心计算：

```
Transformer Block (Qwen2.5-7B)
├── RMSNorm (03_rmsnorm)           ← input_layernorm
├── Self-Attention
│   ├── RoPE (04_rope)             ← 对 Q, K 应用旋转位置编码
│   ├── FlashAttention (05)        ← Q @ K^T → softmax → @ V
│   └── Softmax (02_softmax)       ← attention score 归一化
├── Residual Add + RMSNorm (06)    ← post_attention_layernorm
├── FFN (SwiGLU)
│   └── SiLU activation            ← 可进一步融合
└── Residual Add                   ← 最终残差连接
```

## 面试高频问题速查

### FlashAttention
- **Q**: FlashAttention 为什么快？
- **A**: 核心是 tiling + online softmax。将 Q/K/V 分块加载到 SRAM 中计算，避免将 O(N²) 的 attention matrix 写入 HBM。IO 复杂度从 O(N²d) 降到 O(N²d²/M)。

### Online Softmax
- **Q**: 如何在不知道全局 max 的情况下分块计算 softmax？
- **A**: 维护 running max 和 running sum。遇到新的 max 时，修正之前的 sum：`new_sum = old_sum * exp(old_max - new_max) + exp(x - new_max)`。

### 算子融合
- **Q**: 为什么要做算子融合？
- **A**: LLM 中大量算子是 memory-bound（如 RMSNorm、Add、Activation），瓶颈在显存带宽而非计算。融合减少中间结果的 HBM 读写，直接提升性能。

### RoPE
- **Q**: RoPE 相比绝对位置编码的优势？
- **A**: 旋转后的 Q/K 内积只依赖相对位置差，天然具备相对位置编码能力。且支持长度外推（NTK-aware RoPE、YaRN）。
