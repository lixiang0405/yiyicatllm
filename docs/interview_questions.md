# 🎯 AI Infra 实习面试高频问题 (30 题)

基于本项目的技术栈，按面试频率和重要度排序。每题包含参考答案要点。

---

## 一、CUDA 编程与算子优化 (10 题)

### Q1: GEMM 优化你做了哪些版本？每一步解决了什么问题？

**参考答案**:

做了 6 个版本，逐步优化：

1. **V1 Naive**: 每个线程计算 C 的一个元素，每次从 Global Memory 读 A 和 B。问题是计算访存比极低（~0.25 FLOPs/byte），远低于 GPU 的计算访存比上限。

2. **V2 Shared Memory Tiling**: 将 A 和 B 分块加载到 Shared Memory，block 内线程共享数据。每个元素被复用 TILE_SIZE 次，计算访存比提升到 ~TILE_SIZE/2。

3. **V3 Bank Conflict Padding**: Shared Memory 有 32 个 bank，同一 warp 内线程访问同一 bank 会串行化。解决方法是 padding 列数 +1，错开 bank 映射。

4. **V4 Register Tiling**: 每个线程计算 TM×TN 的子块而非单个元素，用寄存器缓存 A 和 B 的子列/子行，通过外积累加。计算访存比提升到 ~TILE_SIZE × TM × TN / (TM + TN)。

5. **V5 Double Buffering**: 使用两块 Shared Memory 交替使用，在计算当前 tile 的同时预取下一个 tile，隐藏 Global Memory 加载延迟。

6. **V6 Warptiling**: 在 Block Tile 和 Thread Tile 之间加入 Warp Tile 层次，形成 Block→Warp→Thread 三级 tiling。Warp 内线程协作访问 Shared Memory，减少 warp 间的 bank conflict。

---

### Q2: 什么是 Bank Conflict？怎么解决？

**参考答案**:

Shared Memory 被划分为 32 个 bank（每个 bank 宽 4 字节），连续的 4 字节地址映射到连续的 bank。当同一 warp 内的多个线程访问同一 bank 的不同地址时，这些访问会被串行化，称为 bank conflict。

**解决方法**:
- **Padding**: 声明 `__shared__ float A[TILE][TILE+1]`，+1 使得同一列的元素错开 bank
- **Swizzle**: 对地址做异或变换，打乱 bank 映射
- 实际效果：消除 bank conflict 后 Shared Memory 带宽可提升 2-32 倍

---

### Q3: Warp Shuffle 是什么？为什么比 Shared Memory 快？

**参考答案**:

Warp Shuffle（`__shfl_down_sync`、`__shfl_xor_sync` 等）是 warp 内线程直接交换寄存器数据的指令，不经过任何内存。

**比 Shared Memory 快的原因**:
1. 不需要 `__syncthreads()` 同步（warp 内天然同步）
2. 不需要写入再读取 Shared Memory（省去两次内存访问）
3. 延迟更低：~1 cycle vs Shared Memory 的 ~20-30 cycles
4. 不占用 Shared Memory 容量

**典型应用**: Warp-level Reduction（求和、求最大值），在我的 reduce.cu、softmax.cu、rmsnorm.cu 中都有使用。

---

### Q4: 什么是 Roofline Model？如何判断一个 kernel 是 compute-bound 还是 memory-bound？

**参考答案**:

Roofline Model 用算术强度（Arithmetic Intensity = FLOPs / Bytes）来判断性能瓶颈：

- **算术强度 < 拐点**: Memory-bound，优化方向是减少内存访问（向量化、算子融合、缓存优化）
- **算术强度 > 拐点**: Compute-bound，优化方向是提升计算效率（Tensor Core、指令级并行）

**拐点 = 峰值算力 / 峰值带宽**。例如 A100: 19.5 TFLOPS / 2 TB/s ≈ 10 FLOPs/byte。

**实际例子**:
- GEMM (大矩阵): 算术强度 ~O(N)，compute-bound → 用 Tensor Core
- RMSNorm / Softmax: 算术强度 ~O(1)，memory-bound → 用算子融合减少 IO
- GEMV (batch=1): 算术强度 ~O(1)，memory-bound → 用向量化访存

---

### Q5: FlashAttention 的核心思想是什么？

**参考答案**:

FlashAttention 解决的问题：标准 Attention 需要 O(N²) 的显存存储完整的 attention matrix。

**核心思想**:
1. **Tiling**: 将 Q、K、V 分块，每次只处理一个 tile，不需要存储完整的 N×N attention matrix
2. **Online Softmax**: 在分块计算时，用 running max 和 running sum 增量更新 softmax，无需两次遍历
3. **重计算 (Recomputation)**: 反向传播时重新计算 attention matrix 而非存储，用计算换显存

**效果**: 显存从 O(N²) 降到 O(N)，同时因为减少了 HBM 访问，实际速度也更快（2-4x）。

**Online Softmax 公式**:
- 处理新 block 时：更新 max → 用修正因子 `e^(old_max - new_max)` 修正之前的累加结果 → 累加新 block

---

### Q6: 算子融合 (Kernel Fusion) 的原理和收益是什么？

**参考答案**:

**原理**: 将多个独立的 kernel 合并为一个 kernel，减少中间结果对 Global Memory 的读写。

**以 Fused Add + RMSNorm 为例**:
- 非融合: Add kernel (读 x, residual → 写 x') + RMSNorm kernel (读 x' → 写 output) = 4 次 Global Memory IO
- 融合: 一个 kernel 内完成 Add + RMSNorm = 2 次 Global Memory IO
- **IO 量减少 ~40%**

**收益**:
1. 减少 Global Memory 读写（memory-bound 算子的主要瓶颈）
2. 减少 kernel launch 开销（每次 launch ~5-10μs）
3. 中间结果保留在寄存器/Shared Memory 中，延迟更低

**LLM 中常见的融合**: QKV projection 融合、Add+LayerNorm 融合、SwiGLU 激活融合。

---

### Q7: GEMV 和 GEMM 有什么区别？在 LLM 推理中分别对应什么阶段？

**参考答案**:

| | GEMM (M×K × K×N) | GEMV (M×K × K×1) |
|---|---|---|
| **计算类型** | Compute-bound | Memory-bound |
| **算术强度** | O(N) | O(1) |
| **优化方向** | Tiling, Tensor Core | 向量化访存, Warp # LLM 面试题集（50题）

## 一、CUDA 编程与算子优化 (15题)

### Q1: GEMM 优化的 6 个版本演进路线是什么？

**参考答案**:

| 版本 | 优化技术 | 性能提升 |
|-----|---------|---------|
| V1 | 基础实现，每个线程计算一个输出元素 | 基准 |
| V2 | Shared Memory Tiling | ~2x |
| V3 | Register Tiling | ~4x |
| V4 | Bank Conflict 优化 | ~6x |
| V5 | Warptiling (Warp 内并行) | ~8x |
| V6 | 双缓冲 + 向量化 | ~10x |

**核心思想**: 逐步减少对 Global Memory 的访问，让数据尽可能停留在寄存器中。

---

### Q2: 什么是 Bank Conflict？如何避免？

**参考答案**:

**原理**: Shared Memory 分成 32 个 bank，同一 warp 的线程如果访问同一个 bank 的不同地址，会产生序列化访问。

**避免方法**:
1. **Padding**: 在矩阵维度上加 padding，让连续线程访问不同 bank
2. **XOR swap**: 交换访问模式，打乱访问顺序
3. **向量化访问**: 用 `float4` 一次访问 4 个元素，减少 bank 冲突

**示例**: 32×32 矩阵，列访问时每 32 个线程访问同一 bank，加 1 列 padding 即可解决。

---

### Q3: Warp Shuffle 是什么？有什么优势？

**参考答案**:

**原理**: 在 warp 内（32 个线程）直接交换数据，无需通过 Shared Memory。

**优势**:
1. **省去 Shared Memory 访问**: 减少 ~20-30 cycles 延迟
2. **省去 `__syncthreads()`**: 无需同步，减少等待
3. **寄存器间直接传输**: 速度更快

**应用场景**: Reduce 的最后 32 个元素归约、矩阵转置、前缀和等。

---

### Q4: Roofline Model 是什么？如何用它分析性能瓶颈？

**参考答案**:

**原理**: 绘制计算强度（FLOPs/Byte）与理论性能上限的关系图。

**两个关键指标**:
1. **Peak Performance**: 计算峰值（如 A100 ~312 TFLOPS）
2. **Peak Bandwidth**: 内存带宽峰值（如 A100 ~2 TB/s）

**分析方法**:
- 计算算子的 FLOPs/Byte
- 如果低于转折点（~100 FLOPs/Byte），说明是 **Memory-bound**
- 如果高于转折点，说明是 **Compute-bound**

**应用**: GEMM 是 Compute-bound，可以用 Tensor Core；GEMV 是 Memory-bound，优化访存模式。

---

### Q5: FlashAttention 的核心原理是什么？

**参考答案**:

**核心思想**: 将 Attention 计算分块（Tiling），在 SRAM 中分块计算，避免重复读写 HBM。

**关键优化**:
1. **分块计算**: 将 Q、K、V 分成小块，在 SRAM 中计算 Attention
2. **Online Softmax**: 分块计算 Softmax，避免存储完整的注意力矩阵
3. **重计算**: 反向传播时重算中间结果，节省显存

**效果**: 显存占用从 O(N²) 降到 O(N)，速度提升 2-4 倍。

---

### Q6: Online Softmax 是如何实现的？

**参考答案**:

**问题**: 传统 Softmax 需要先计算所有 token 的 exp 值，然后归一化，需要存储完整矩阵。

**Online Softmax**:
1. **分块计算**: 每个块单独计算 max 和 sum
2. **增量更新**: 用当前块的 max 和 sum 更新全局的 max 和 sum
3. **无需存储完整矩阵**: 只需要维护全局的 max 和 sum

**公式**:
```
global_max = max(global_max, block_max)
global_sum = global_sum * exp(block_max - global_max) + block_sum
```

---

### Q7: 什么是算子融合？有什么好处？

**参考答案**:

**定义**: 将多个独立的算子合并成一个 kernel，减少中间结果的存储和读取。

**好处**:
1. **减少 HBM 访问**: 中间结果留在寄存器/Shared Memory
2. **减少 kernel launch 开销**: 多个操作合并，减少调度成本
3. **提升缓存利用率**: 数据复用更好

**示例**: LayerNorm → Activation → Linear 可以融合成一个 kernel。

---

### Q8: GEMV 和 GEMM 的区别是什么？为什么 GEMV 更难优化？

**参考答案**:

| | GEMM | GEMV |
|---|---|---|
| 输入 | 矩阵 × 矩阵 | 矩阵 × 向量 |
| 计算模式 | 大规模并行 | 受限并行 |
| 瓶颈 | Compute-bound | Memory-bound |
| 优化难度 | 低 | 高 |

**GEMV 更难优化的原因**:
1. 每个线程计算一个输出，并行度低
2. 访存模式不规则，难以利用向量化
3. 计算访存比低，内存带宽是瓶颈

**优化方法**: 向量化加载、Shared Memory 缓存、多轮迭代。

---

### Q9: float4 向量化访存的原理和限制是什么？

**参考答案**:

**原理**: GPU 的 Global Memory 事务以 32/64/128 字节为单位。`float4` 一次加载 4 个 float (16 字节)，比逐个 float 加载减少 4 倍的内存事务数。

**限制**:
1. **对齐要求**: 地址必须 16 字节对齐，否则会拆分为多次事务
2. **数组长度**: 必须是 4 的倍数，否则需要处理尾部元素
3. **不适合随机访问**: 向量化适合连续访问模式

---

### Q10: GPU 的内存层次结构是什么？

**参考答案**:

```
寄存器 (Register)     → ~0 cycle,  最快，每线程私有
    ↓
Shared Memory / L1    → ~20-30 cycles, ~19 TB/s (A100)
    ↓
L2 Cache              → ~200 cycles, ~6 TB/s
    ↓
Global Memory (HBM)   → ~400-600 cycles, ~2 TB/s (A100 HBM2e)
```

**优化原则**: 尽量让数据停留在高层（寄存器 > Shared Memory > L2 > HBM）。

---

### Q11: Reduce 并行归约有哪些优化技巧？

**参考答案**:

1. **避免 Warp Divergence**: 使用连续线程参与归约（stride 从大到小）
2. **Warp Shuffle**: 最后 32 个元素用 `__shfl_down_sync` 省去 Shared Memory
3. **Grid-stride Loop**: 一个 block 处理多个数据段
4. **向量化加载**: 用 `float4` 一次加载 4 个元素
5. **两阶段归约**: Block 内归约 → Block 间用 atomicAdd 汇总

---

### Q12: RMSNorm 的计算公式是什么？和 LayerNorm 的区别？

**参考答案**:

**RMSNorm 公式**:
```
output = x / sqrt(mean(x²) + ε) * γ
```

**与 LayerNorm 的区别**:
- LayerNorm: `(x - mean) / std * γ + β`（中心化）
- RMSNorm: `x / sqrt(mean(x²)) * γ`（无中心化）

**优势**:
- 计算更快（无需减去均值）
- 效果相当甚至更好
- LLaMA、Qwen 等现代模型都使用 RMSNorm

---

### Q13: RoPE (Rotary Position Embedding) 的原理是什么？

**参考答案**:

**核心思想**: 通过旋转矩阵给 query 和 key 注入相对位置信息。

**公式**:
```
q'_m = R(θ, m) * q_m
k'_n = R(θ, n) * k_n
attention(q'_m, k'_n) = f(q_m, k_n, m - n)
```

**优势**:
1. 相对位置编码，外推性好
2. 无需额外训练位置编码
3. 计算高效（可以用复数乘法）

---

### Q14: Shared Memory 的特点和用途是什么？

**参考答案**:

**特点**:
1. **速度快**: ~20-30 cycles，比 HBM 快 10-20 倍
2. **可编程**: 程序员可以控制数据加载
3. **Block 内共享**: 同一个 block 的线程可以访问

**用途**:
1. **Tiling**: 缓存全局内存的数据
2. **Reduce**: 线程间数据交换
3. **转置**: 矩阵转置优化

**限制**: 每个 SM 只有几十 KB，需要谨慎使用。

---

### Q15: Thread、Block、Grid 的层次关系是什么？

**参考答案**:

**层次结构**:
```
Grid (整个 kernel)
  ├─ Block (共享 Shared Memory)
  │   ├─ Warp (32 个线程，同步执行)
  │   │   └─ Thread (最小执行单元)
```

**调度规则**:
1. **Thread**: 执行基本的计算任务
2. **Warp**: 32 个线程，SIMD 执行
3. **Block**: 可以同步（`__syncthreads()`），共享 Shared Memory
4. **Grid**: 包含多个 Block，由 GPU 调度

**配置**: `<<<grid_dim, block_dim>>>`

---

## 二、LLM 训练与微调 (12题)

### Q16: LoRA 的原理是什么？为什么有效？

**参考答案**:

**原理**: 冻结预训练权重 W，在旁边加一个低秩分解 ΔW = A × B（A: d×r, B: r×d, r << d）。前向传播时 output = Wx + ABx。

**为什么有效**:
1. 预训练模型的权重更新矩阵是低秩的
2. rank=16 时只训练 ~0.1% 参数，但效果接近全参数微调
3. 推理时可以将 AB 合并回 W，无额外推理开销

**关键超参数**:
- `rank`: 秩，通常 8-64
- `alpha`: 缩放系数，通常 alpha = 2 × rank
- `target_modules`: 应用 LoRA 的层

---

### Q17: QLoRA 和 LoRA 的区别是什么？

**参考答案**:

| | LoRA | QLoRA |
|---|---|---|
| 量化 | 不量化 | 4-bit 量化 |
| 显存占用 | 较高 | **更低** |
| 训练速度 | 快 | 稍慢 |
| 精度 | 高 | 略低 |

**QLoRA 的关键创新**:
1. **4-bit NormalFloat**: 适合正态分布的量化方式
2. **Double Quantization**: 对量化参数也进行量化
3. **Paged Optimizers**: 用 CPU 内存处理优化器状态

---

### Q18: DeepSpeed ZeRO 的三个 Stage 分别做了什么？

**参考答案**:

| Stage | 分片内容 | 显存节省 | 通信开销 |
|-------|---------|---------|---------|
| ZeRO-1 | 优化器状态 | ~4x | 低 |
| ZeRO-2 | 优化器状态 + 梯度 | ~8x | 中 |
| ZeRO-3 | 优化器状态 + 梯度 + 模型参数 | ~N×(N=GPU数) | 高 |

**选择建议**:
- 7B 模型用 ZeRO-2 足够
- 30B+ 模型用 ZeRO-3

---

### Q19: DPO 的原理和 Loss 是什么？

**参考答案**:

**原理**: 直接用偏好数据优化策略，跳过 Reward Model 训练。数学上证明：最优策略和最优 Reward Model 之间存在解析映射。

**Loss**:
```
L = -log(sigmoid(β × (log π(chosen)/ref(chosen) - log π(rejected)/ref(rejected))))
```

**优势**:
- 只需要 2 个模型（Policy, Reference）
- 训练复杂度低
- 不需要调 PPO 的超参数

---

### Q20: GRPO 的原理是什么？和 PPO 有什么区别？

**参考答案**:

**GRPO (Group Relative Policy Optimization)** 是 DeepSeek 提出的 RL 训练方法。

**与 PPO 的区别**:
- PPO: 需要 Critic Model 估计 baseline
- GRPO: 用同一 prompt 的多个采样结果的平均 reward 作为 baseline

**优势**:
- 省去 Critic Model，减少 ~25% 的 GPU 显存
- 训练更稳定（组内相对比较比绝对值更鲁棒）

---

### Q21: PPO、DPO、GRPO 的对比？

**参考答案**:

| | PPO | DPO | GRPO |
|---|---|---|---|
| 需要 Reward Model | ✅ | ❌ | ✅ |
| 需要 Critic Model | ✅ | ❌ | ❌ |
| 模型数量 | 4 | 2 | 3 |
| 训练复杂度 | 高 | 低 | 中 |
| 稳定性 | 中 | 高 | 高 |
| 适用场景 | 大规模数据 | 小规模数据 | 推理任务 |

---

### Q22: veRL 框架是什么？有什么特点？

**参考答案**:

**veRL (Versatile Reinforcement Learning)** 是 DeepSeek 开发的 RL 训练框架。

**特点**:
1. **支持 GRPO**: 专门优化了 GRPO 算法
2. **高效通信**: 优化的多卡通信机制
3. **易用性**: 简洁的 API，易于集成
4. **兼容性**: 支持 HuggingFace 模型

---

### Q23: 梯度累积的作用是什么？

**参考答案**:

**作用**: 在显存不足以支撑大 batch size 时，通过多次前向+反向传播累积梯度，等效于更大的 batch size。

**原理**:
- 设 `per_device_batch_size=2`, `gradient_accumulation_steps=8`, `num_gpus=4`
- 等效 batch size = 2 × 8 × 4 = 64
- 每 8 步才做一次参数更新

---

### Q24: BF16 和 FP16 的区别是什么？

**参考答案**:

| | FP16 | BF16 |
|---|---|---|
| 符号位 | 1 | 1 |
| 指数位 | 5 | **8** |
| 尾数位 | **10** | 7 |
| 数值范围 | ±65504 | ±3.4×10³⁸ (同 FP32) |
| 精度 | 更高 | 较低 |

**推荐 BF16**: 数值范围与 FP32 相同，不容易溢出，不需要 loss scaling。

---

### Q25: 混合精度训练的原理是什么？

**参考答案**:

**原理**: 前向传播和反向传播用 FP16/BF16，优化器状态用 FP32。

**好处**:
1. **减少显存**: FP16 显存占用是 FP32 的一半
2. **加速训练**: FP16 计算更快（Tensor Core）
3. **保持精度**: FP32 优化器状态保证数值稳定性

**关键**: 需要动态 loss scaling（FP16）或使用 BF16（无需 scaling）。

---

### Q26: 学习率调度有哪些常见策略？

**参考答案**:

| 策略 | 特点 | 适用场景 |
|-----|------|---------|
| Constant | 学习率不变 | 简单任务 |
| Linear Decay | 线性衰减 | 预训练 |
| Cosine Decay | 余弦衰减 | 微调 |
| Warmup + Decay | 先 warmup 再衰减 | 大多数场景 |

**LoRA 微调**: 通常用 `cosine` 或 `linear`，学习率比预训练大 10-100 倍。

---

### Q27: 数据工程在 LLM 训练中的重要性？

**参考答案**:

**核心观点**: "Garbage in, Garbage out"

**关键环节**:
1. **数据清洗**: 去除低质量、重复数据
2. **数据增强**: 回译、改写、合成数据
3. **数据平衡**: 保证各领域数据比例合理
4. **指令微调**: 构造高质量的 instruction-following 数据

**经验**: 高质量数据比大模型更重要。

---

## 三、模型量化 (6题)

### Q28: AWQ 和 GPTQ 的区别是什么？

**参考答案**:

| | AWQ | GPTQ |
|---|---|---|
| 全称 | Activation-aware Weight Quantization | Generalized Post-Training Quantization |
| 核心思想 | 根据激活值分布保护重要通道 | 逐层最小化量化误差 |
| 量化速度 | **快**（~10 分钟） | 较慢（~30-60 分钟） |
| 精度 | 略优 | 好 |

**AWQ 的关键**: 保护 1% 的显著权重通道。

---

### Q29: INT4 和 INT8 量化的原理是什么？

**参考答案**:

**基本原理**: 将 FP16 权重映射到整数范围。

**INT4**: 映射到 [0, 15]
```
x_int = round((x_fp - zero_point) / scale)
```

**INT8**: 映射到 [-128, 127]

**Group Quantization**: 每 128 个元素一组，每组有自己的 scale 和 zero_point。

---

### Q30: Group Quantization 的优势是什么？

**参考答案**:

**优势**:
1. **适应局部分布**: 不同区域的数值分布不同，分组量化能更好地适应
2. **精度提升**: 比整张 tensor 用一组 scale 精度更高
3. **存储开销小**: group_size=128 时，额外存储开销 <1%

**常用值**: group_size=128（平衡精度和存储）。

---

### Q31: 如何评估量化精度损失？

**参考答案**:

**评估方法**:
1. **Perplexity**: 在验证集上计算困惑度，越低越好
2. **下游任务**: 在 MMLU、C-Eval 等 benchmark 上对比
3. **人工评估**: 对比量化前后的回答质量

**预期损失**:
- INT4: perplexity 增加 0.1-0.5，准确率下降 <1%
- INT8: 几乎无损失

---

### Q32: 量化对推理速度的影响是什么？

**参考答案**:

**影响**:
1. **显存占用**: INT4 比 FP16 减少 ~75%
2. **计算速度**: INT4 计算更快（如果硬件支持）
3. **带宽瓶颈**: INT4 加载更快，但解码可能仍受限于内存带宽

**实际效果**:
- INT4: 吞吐量提升 1.5-2x
- INT8: 吞吐量提升 1.2-1.5x

---

### Q33: SmoothQuant 的原理是什么？

**参考答案**:

**原理**: 在量化前对权重进行平滑处理，平衡激活值和权重的量化难度。

**公式**:
```
Y = (X / s) * (W * s)
```
- `s`: 每通道的平滑因子
- 激活值除以 s，权重乘以 s

**效果**: 减少量化误差，提升 INT8 精度。

---

## 四、推理部署与优化 (10题)

### Q34: vLLM 的 PagedAttention 是什么？

**参考答案**:

**问题**: KV Cache 传统预分配方式导致 60-80% 显存浪费。

**解决方案**:
1. 借鉴操作系统的虚拟内存分页机制
2. 将 KV Cache 分成固定大小的 block（如 16 个 token）
3. 用 block table 维护逻辑→物理的映射
4. 按需分配 block，不需要连续内存

**效果**: 显存利用率从 ~20-40% 提升到 ~90%+。

---

### Q35: Continuous Batching 和 Static Batching 的区别？

**参考答案**:

| | Static Batching | Continuous Batching |
|---|---|---|
| 批处理方式 | 一批请求一起开始 | 每个 decode step 都可以插入/移除请求 |
| GPU 利用率 | 低（短请求等待长请求） | 高（始终满载） |
| 吞吐量 | 低 | **高（2-3x）** |

**Continuous Batching**: 也叫 Iteration-level Batching。

---

### Q36: KV Cache 的原理和显存计算？

**参考答案**:

**原理**: 在自回归生成时，缓存之前 token 的 Key 和 Value 向量，避免重复计算。

**显存计算**:
- 每层每 token: 2 (K+V) × hidden_size × 2 bytes (FP16)
- Qwen2.5-7B (32 层, hidden=4096): 每 token = 2 × 32 × 4096 × 2 = 512 KB
- 2048 tokens: 512 KB × 2048 = 1 GB

---

### Q37: Prefill 和 Decode 阶段的区别？

**参考答案**:

| | Prefill | Decode |
|---|---|---|
| 输入 | 整个 prompt (N tokens) | 1 个 token |
| 计算模式 | GEMM (矩阵×矩阵) | GEMV (矩阵×向量) |
| 瓶颈 | Compute-bound | Memory-bound |
| 耗时 | 短（一次性） | 长（逐 token 生成） |

**关键指标**:
- **TTFT**: Time To First Token（首字延迟）
- **TPOT**: Time Per Output Token（每 token 耗时）
- **Throughput**: 单位时间生成的 token 数

---

### Q38: Prefix Caching 的原理和适用场景？

**参考答案**:

**原理**: 多个请求共享相同的 system prompt 时，缓存这部分的 KV Cache。

**适用场景**:
1. 所有请求共享相同的 system prompt
2. 多轮对话中，历史消息部分不变
3. RAG 场景中，检索到的文档片段可能重复

**效果**: 如果 system prompt 有 500 tokens，100 个请求可以节省 99 × 500 tokens 的 Prefill 计算。

---

### Q39: Speculative Decoding 的原理是什么？

**参考答案**:

**原理**: 用一个小模型快速生成多个 token，然后用大模型验证。

**流程**:
1. 小模型生成 k 个候选 token
2. 大模型并行验证这 k 个 token
3. 接受验证通过的 token，拒绝的重新生成

**效果**: 吞吐量提升 1.5-2x，尤其适合低延迟场景。

---

### Q40: GQA 和 MQA 的区别是什么？

**参考答案**:

| | MHA | MQA | GQA |
|---|---|---|---|
| KV Head 数 | N | 1 | N/group_size |
| 显存占用 | 高 | 低 | 中 |
| 性能 | 低 | 高 | 高 |

**MQA (Multi-Query Attention)**: 所有 query head 共享一个 KV head。

**GQA (Grouped-Query Attention)**: query head 分组，每组共享一个 KV head。

**优势**: 减少 KV Cache 显存占用，提升推理速度。

---

### Q41: Tensor Parallelism 和 Pipeline Parallelism 的区别？

**参考答案**:

| | Tensor Parallelism (TP) | Pipeline Parallelism (PP) |
|---|---|---|
| 切分方式 | 每层的权重矩阵切分到多卡 | 不同层分配到不同卡 |
| 通信 | 每层都需要 AllReduce | 只在层边界传递激活值 |
| 延迟 | 低（层内并行） | 高（有 bubble） |
| 适用场景 | 同一节点内多卡 | 跨节点 |

**推理时通常用 TP**: 因为推理对延迟敏感。

---

### Q42: 推理性能指标有哪些？

**参考答案**:

| 指标 | 全称 | 含义 |
|-----|------|------|
| TTFT | Time To First Token | 首字延迟（用户体验） |
| TPOT | Time Per Output Token | 每生成一个 token 的耗时 |
| Throughput | 吞吐量 | 单位时间生成的 token 数 |
| Latency | 延迟 | 请求的总耗时 |

**优化目标**: 降低 TTFT 和 TPOT，提升 Throughput。

---

### Q43: 常见的 Attention 变体有哪些？

**参考答案**:

| 变体 | 特点 | 应用 |
|-----|------|------|
| FlashAttention | 分块计算，显存优化 | 训练和推理 |
| Sliding Window | 只关注局部窗口 | 长文本 |
| Sparse Attention | 稀疏注意力 | 长文本 |
| Linear Attention | 线性复杂度 | 长文本 |
| GQA/MQA | 减少 KV Cache | 推理 |

---

## 五、Transformer 与 LLM 基础 (5题)

### Q44: Transformer 的核心架构是什么？

**参考答案**:

**核心组件**:
1. **Self-Attention**: 捕捉序列内依赖关系
2. **Multi-Head Attention**: 多头并行计算
3. **Feed-Forward Network**: 非线性变换
4. **LayerNorm**: 归一化
5. **Residual Connection**: 残差连接

**优势**:
- 并行计算（相比 RNN）
- 长距离依赖建模
- 可扩展性强

---

### Q45: Self-Attention 的计算复杂度是多少？

**参考答案**:

**复杂度**:
- **时间复杂度**: O(N² × d)（N 是序列长度，d 是 hidden size）
- **空间复杂度**: O(N²)（注意力矩阵）

**瓶颈**: N² 复杂度限制了长文本处理。

**优化方法**:
- FlashAttention: O(N)
- Sliding Window: O(N × window_size)
- Linear Attention: O(N)

---

### Q46: RoPE 和 ALiBi 的区别是什么？

**参考答案**:

| | RoPE | ALiBi |
|---|---|---|
| 类型 | 相对位置编码 | 相对位置编码 |
| 外推性 | 好 | 好 |
| 实现方式 | 旋转矩阵 | 偏置项 |
| 训练 | 需要训练位置编码 | 无需训练 |
| 应用 | LLaMA, Qwen | BLOOM |

**RoPE**: 通过旋转矩阵注入相对位置信息。
**ALiBi**: 在注意力分数上减去距离偏置。

---

### Q47: SwiGLU 激活函数是什么？

**参考答案**:

**公式**:
```
SwiGLU(x) = Swish(xW) ⊙ (xV)
```
- `Swish(x) = x * sigmoid(x)`
- `⊙`: 元素乘

**优势**:
1. 比 ReLU 效果更好
2. 平滑性好
3. LLaMA、Qwen 等现代模型都使用

---

### Q48: LLM 的采样策略有哪些？

**参考答案**:

| 策略 | 原理 | 效果 |
|-----|------|------|
| Greedy | 选择概率最高的 token | 确定性，但可能重复 |
| Temperature | 调整概率分布平滑度 | 温度越高越随机 |
| Top-k | 从概率最高的 k 个 token 中采样 | 避免低概率 token |
| Top-p (Nucleus) | 从累积概率达到 p 的 token 中采样 | 动态调整候选集 |

**常用组合**: Temperature + Top-p

---

## 六、综合与系统设计 (2题)

### Q49: 如何设计一个大模型部署方案？

**参考答案**:

**考虑因素**:
1. **模型大小**: 7B 用单卡，70B 用多卡或量化
2. **延迟要求**: 低延迟用 TP，高吞吐用 PP
3. **并发量**: 用 vLLM 的 PagedAttention + Continuous Batching
4. **成本**: 量化减少显存占用

**方案示例**:
- 70B 模型：INT4 量化 + 2 张 A100 (TP=2) + vLLM
- 7B 模型：FP16 + 单张 3090 + vLLM

---

### Q50: 你对 AI Infra 的理解是什么？

**参考答案**:

AI Infra 是连接算法和硬件的桥梁，核心目标是**让模型在有限的硬件资源上跑得更快、更省**。

**具体包括**:
1. **训练侧**: 分布式训练框架（DeepSpeed, Megatron-LM）、混合精度、梯度检查点、通信优化
2. **推理侧**: 推理引擎（vLLM, TensorRT-LLM）、量化、KV Cache 管理、调度策略
3. **算子层**: CUDA kernel 优化、算子融合、Tensor Core 利用
4. **系统层**: GPU 集群调度、显存管理、多模型服务

**我的项目覆盖**: 训练（DeepSpeed + LoRA + DPO + GRPO）、推理（vLLM + 量化）、算子（CUDA C++ 手写优化）。
