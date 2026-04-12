# 📄 简历 - AI Infra 实习生

---

## 个人技能

- **LLM 训练**: 熟悉 LoRA/QLoRA 参数高效微调原理与实践，掌握 DPO/GRPO 等偏好对齐方法，了解 PPO 训练流程，有 veRL GRPO 强化学习训练实战经验
- **分布式训练**: 熟悉 DeepSpeed ZeRO (Stage 1/2/3) 的分片策略，理解数据并行、模型并行、流水线并行的原理与适用场景
- **模型量化与部署**: 掌握 GPTQ/GGUF 量化原理与实践，熟悉 llama.cpp 本地部署方案，理解量化对精度和推理性能的影响
- **推理优化**: 熟悉 vLLM 推理引擎核心技术（PagedAttention、Continuous Batching、Tensor Parallel），理解 KV Cache 管理机制
- **CUDA 编程**: 使用 CUDA C++ 和 Triton 实现 LLM 核心算子（GEMM、Softmax、FlashAttention、RMSNorm 等），掌握 Shared Memory Tiling、Bank Conflict 优化、Register Tiling、Warp Shuffle、向量化访存、算子融合等优化技术
- **编程语言**: Python, C/C++, CUDA C++
- **工具框架**: PyTorch, Transformers, PEFT, LLaMA-Factory, DeepSpeed, vLLM, Triton, veRL

---

## 项目经历

### 基于 Qwen2.5-7B 的中科大智能问答助手 — LLM 全链路训推项目

**项目简介**: 独立完成大语言模型从数据工程、分布式训练、偏好对齐、模型量化到推理部署的完整生产链路，DPO 对齐后 ROUGE-L 提升 30.7%、关键词命中率提升 117.6%，并使用 CUDA C++ / Triton 手写实现 LLM 核心算子。

**技术栈**: Qwen2.5-7B / LoRA / DeepSpeed ZeRO-2 / DPO / GRPO / GPTQ INT4 / GGUF / vLLM / llama.cpp / CUDA C++ / Triton

#### 一、数据工程

- 爬取中科大官网、课程仓库、百度百科等数据源，使用 GPT-4o-mini API 批量生成高质量 QA 对，构建 **6258 条** Alpaca 格式训练数据集
- 设计数据切分策略：从偏好数据中切分公共验证集，确保 SFT/DPO/GRPO 三阶段使用同一验证集进行公平对比
- 构建 DPO 偏好数据集：使用 SFT 模型通过 **vLLM 双卡 TP=2 并行推理**批量生成 rejected 回答，与人工标注的 chosen 组成偏好对
- 实现数据清洗、去重、格式校验等预处理流程，合并 SFT 数据（new_qa + preference chosen）避免灾难性遗忘

#### 二、训练与对齐 (SFT → DPO → GRPO)

- **SFT 微调**: 基于 LLaMA-Factory 框架，使用 **LoRA (rank=64, alpha=128, target=all)** 对 Qwen2.5-7B 全部线性层进行微调，batch_size=16，cosine 学习率调度，训练 6 个 epoch，ROUGE-L 从基座的 0.2099 提升至 0.2415（**+15.1%**），关键词命中率提升 **92.2%**
- **分布式训练**: 配置 **DeepSpeed ZeRO-2** 实现双卡分布式训练，将优化器状态和梯度分片到 2 张 RTX 5090 (32GB)，显著降低单卡显存占用
- **DPO 偏好对齐**: 在 SFT 合并模型基础上使用 **DPO (beta=0.1, sigmoid loss)** 进行偏好对齐，lr=1e-5，训练 1 个 epoch 避免过拟合，ROUGE-L 进一步提升至 0.2743（相对基座 **+30.7%**），关键词命中率达 0.4334（**+117.6%**），事实命中率达 0.7271（**+16.9%**）
- **GRPO 强化学习**: 使用自研 GRPO 训练脚本（基于 veRL 思想），设计多维度规则奖励函数（格式、关键词命中、ROUGE-L 相似度、事实命中率、相对长度系数），采用乘法结构归一化奖励信号。实验发现 GRPO（ROUGE-L 0.27）略低于 DPO（0.2743），分析原因为小数据集下在线 RL 的探索效率不足，最终选择 DPO 模型作为最优产出

#### 三、模型量化

- 使用 **GPTQ INT4** 量化（group_size=128, desc_act=True），从训练数据中采样 128 条作为校准集，确保校准分布与训练分布一致
- 导出 **GGUF Q8_0 格式**（7.5GB），支持 llama.cpp 本地部署，可进一步量化为 Q4_K_M（~4.4GB）适配 8GB 显存消费级 GPU

#### 四、推理部署

- 训练阶段使用 **vLLM** 进行高速推理（~6900 tok/s），利用 **Tensor Parallel (TP=2)** 双卡并行加速 rejected 生成和模型评测
- 部署阶段使用 **llama.cpp** 在 RTX 5070 Laptop (8GB) 上实现 GGUF 量化模型的本地推理，提供 OpenAI 兼容的 REST API
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

1. **全链路覆盖**: 数据工程 → SFT → DPO → GRPO → 量化 → 部署，独立完成完整生产链路
2. **量化的评测数据**: DPO 对齐后 ROUGE-L +30.7%、关键词命中率 +117.6%、事实命中率 +16.9%，有完整的多模型对比评测报告
3. **对齐方法对比**: 实践了 SFT → DPO → GRPO 三种对齐方法，通过实验数据分析了 DPO 在小数据集场景下优于 GRPO 的原因（离线偏好学习 vs 在线探索的效率差异）
4. **CUDA 算子深度**: 不是调库，而是从 Naive 到 Warptiling 逐步优化，理解每一步的性能瓶颈
5. **工程落地**: 在 8GB 显存的消费级 GPU 上跑通 GGUF 量化推理，提供 OpenAI 兼容 API，可实际对外服务
