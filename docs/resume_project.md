# 📄 简历 - AI Infra 实习生

---

## 个人技能

- **LLM 训练**: 熟悉 LoRA/QLoRA 参数高效微调原理与实践，掌握 DPO/GRPO 等偏好对齐方法，了解 PPO 训练流程，熟悉 veRL GRPO 强化学习训练
- **分布式训练**: 熟悉 DeepSpeed ZeRO (Stage 1/2/3) 的分片策略，理解数据并行、模型并行、流水线并行的原理与适用场景
- **模型量化**: 掌握 AWQ/GPTQ INT4 量化原理，理解量化对精度和推理性能的影响
- **推理优化**: 熟悉 vLLM 推理引擎核心技术（PagedAttention、Continuous Batching、Prefix Caching），理解 KV Cache 管理机制
- **CUDA 编程**: 使用 CUDA C++ 和 Triton 实现 LLM 核心算子（GEMM、Softmax、FlashAttention、RMSNorm 等），掌握 Shared Memory Tiling、Bank Conflict 优化、Register Tiling、Warp Shuffle、向量化访存、算子融合等优化技术
- **编程语言**: Python, C/C++, CUDA C++
- **工具框架**: PyTorch, Transformers, PEFT, LLaMA-Factory, DeepSpeed, vLLM, Triton, veRL

---

## 项目经历

### 基于 Qwen2.5-7B 的中科大智能问答助手 — LLM 全链路训推项目

**项目简介**: 独立完成大语言模型从数据工程、分布式训练、偏好对齐、模型量化到推理部署的完整生产链路，并使用 CUDA C++ / Triton 手写实现 LLM 核心算子。

**技术栈**: Qwen2.5-7B / LoRA / DeepSpeed ZeRO-2 / DPO / AWQ INT4 / vLLM / CUDA C++ / Triton

#### 一、数据工程

- 爬取中科大官网、百度百科等数据源，使用 GPT-4o-mini API 批量生成高质量 QA 对，构建 Alpaca 格式训练数据集
- 构建 DPO 偏好数据集（chosen/rejected 对），用于后续偏好对齐训练
- 实现数据清洗、去重、格式校验等预处理流程

#### 二、训练与对齐 (SFT + DPO + GRPO)

- 基于 LLaMA-Factory 框架，使用 **LoRA (rank=16, alpha=32)** 对 Qwen2.5-7B 进行 SFT 微调，仅训练 ~0.1% 参数
- 配置 **DeepSpeed ZeRO-2** 实现多卡分布式训练，将优化器状态和梯度分片到多张 GPU，单卡显存占用从 ~28GB 降至 ~16GB
- SFT 之后使用 **DPO (Direct Preference Optimization)** 进行偏好对齐，beta=0.1，直接用偏好数据优化策略，无需单独训练 Reward Model
- 使用 **veRL 框架进行 GRPO 强化学习训练**，GRPO 不需要 Critic Model，用组内相对排名作为 baseline，设计规则奖励函数（长度、格式、关键词、流畅度），全链路覆盖三种对齐方法：SFT → DPO → GRPO

#### 三、模型量化

- 使用 **AWQ (Activation-aware Weight Quantization)** 将 FP16 模型量化为 INT4，模型体积从 ~14GB 压缩至 ~4.5GB
- 同时支持 GPTQ 量化方案，AWQ 基于激活值分布保护显著权重通道，量化后精度损失 <1%

#### 四、推理部署

- 使用 **vLLM** 部署推理服务，利用 PagedAttention 实现 KV Cache 的分页管理，显存利用率提升 ~60%
- 开启 Continuous Batching 支持动态请求合并，吞吐量相比静态 batching 提升 2-3 倍
- 在 RTX 5070 Laptop (8GB) 上实现 INT4 量化模型的本地推理，TTFT < 200ms
- 基于 Gradio 构建 Web 聊天界面，编写性能 Benchmark 脚本测试 TTFT、吞吐量、并发性能

#### 五、CUDA 算子实现 (13 个文件, 23+ kernel 版本)

**GEMM 矩阵乘法** (6 个优化版本，面试最高频):
- V1 Naive → V2 Shared Memory Tiling (数据复用提升 TILE_SIZE 倍) → V3 Bank Conflict Padding (+1 错开 bank) → V4 Register Tiling (每线程计算 TM×TN 子块，外积累加) → V5 Double Buffering (预取下一 tile 隐藏加载延迟) → V6 Warptiling (Block→Warp→Thread 三级 tiling)

**其他核心算子**:
- **Reduce 并行归约** (4 版本): Naive → Warp Shuffle (`__shfl_down_sync`) → float4 向量化 + Max Reduce
- **GEMV 矩阵向量乘** (4 版本): 行并行 → Warp 协作 → float4 向量化 → Shared Memory 缓存向量
- **Softmax** (2 版本): Safe Softmax + Online Softmax (FlashAttention 的数学基础)
- **RMSNorm** (2 版本): 基础版 + float4 向量化访存
- **RoPE 旋转位置编码**: 复数旋转实现，支持动态序列长度
- **FlashAttention**: Tiling + Online Softmax，O(N) 显存复杂度
- **算子融合 Fused Add+RMSNorm**: 将 2 次 kernel launch 合并为 1 次，IO 量减少 ~40%

---

## 项目亮点（面试时重点强调）

1. **全链路覆盖**: 数据 → SFT → DPO → 量化 → 部署，不是只做了某一环
2. **CUDA 算子深度**: 不是调库，而是从 Naive 到 Warptiling 逐步优化，理解每一步的性能瓶颈
3. **工程落地**: 在 8GB 显存的消费级 GPU 上跑通量化推理，有实际的性能数据
4. **对齐技术**: 不只做 SFT，还做了 DPO 偏好对齐和 GRPO 强化学习，全链路覆盖三种对齐方法：SFT → DPO → GRPO，理解 DPO vs PPO vs GRPO 的原理差异
