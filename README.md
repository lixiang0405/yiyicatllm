# 🎓 中科大智能问答助手 (USTC-QA)

基于 **Qwen2.5-7B + LoRA 微调 + DeepSpeed 分布式训练 + GGUF 量化 + llama.cpp 部署** 的 LLM 全链路训推项目。

## 项目概述

本项目展示了大语言模型从数据工程、分布式训练、模型量化到推理部署的**完整生产链路**：

```
数据爬取/生成 → SFT 微调(DeepSpeed) → DPO 对齐 → GRPO 强化学习 → 权重合并 → GGUF 量化 → llama.cpp 本地部署
```

### 技术栈

| 环节 | 技术方案 |
|------|----------|
| 基座模型 | Qwen2.5-7B |
| 微调方法 | LoRA (PEFT) |
| 训练框架 | LLaMA-Factory + DeepSpeed ZeRO-2 |
| 分布式训练 | 2 × RTX 5090 + DeepSpeed |
| 偏好对齐 | DPO (Direct Preference Optimization) |
| 强化学习 | veRL GRPO (规则奖励函数) |
| 模型量化 | GGUF Q8_0 (llama.cpp) |
| 推理部署 | llama.cpp (OpenAI 兼容 API) |

### 项目结构

```
yiyicat-llm/
├── README.md                       # 项目文档
├── requirements.txt                # Python 依赖
├── .gitignore
├── scripts/
│   └── setup_env.sh                # 环境搭建脚本
├── data/
│   ├── sample_data.json            # 示例 QA 数据 (8条)
│   ├── prepare_data.py             # 数据爬取与预处理
│   └── generate_qa.py              # 用大模型批量生成 QA 对
├── train/
│   ├── train_lora.yaml             # LLaMA-Factory 训练配置
│   ├── ds_config_zero2.json        # DeepSpeed ZeRO-2 配置
│   ├── run_train.sh                # 训练启动脚本 (支持单卡/多卡)
│   ├── verl_config.yaml            # veRL GRPO 配置
│   ├── reward_function.py          # 规则奖励函数
│   ├── run_grpo.sh                 # GRPO 训练启动脚本
│   ├── prepare_grpo_data.py        # GRPO 数据准备
│   └── merge_lora.py               # LoRA 权重合并
├── quantize/
│   └── quantize_model.py           # 模型量化 (GPTQ/GGUF)
├── deploy/
│   ├── serve.sh                    # llama.cpp 推理服务启动
│   └── chat_demo.py                # Gradio 聊天界面
└── benchmark/
    └── benchmark.py                # 推理性能测试
```

---

## 环境要求

| 项目 | 版本 |
|------|------|
| OS | Windows 11 WSL2 / Ubuntu |
| Python | 3.11 |
| CUDA | 12.8 |
| PyTorch | 2.10.0+cu128 |
| GPU (训练) | RTX 5090 × 2 |
| GPU (推理) | RTX 5070 Laptop 8GB |

---

## 快速开始

### Step 0: 环境搭建

```bash
# 克隆项目
git clone https://github.com/yiyicat/yiyicat-llm.git
cd yiyicat-llm

# 运行环境搭建脚本
bash scripts/setup_env.sh

# 或手动安装
conda create -n yiyicat-llm python=3.11 -y
conda activate yiyicat-llm
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# 安装 LLaMA-Factory
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory && pip install -e ".[torch,metrics]" && cd ..
```

### Step 1: 数据准备

#### 方式 A：使用示例数据（快速体验）

项目已包含 8 条手写的中科大 QA 数据 (`data/sample_data.json`)，可直接用于训练。

```bash
cp data/sample_data.json data/train_data.json
```

#### 方式 B：爬取 + 生成完整数据集（推荐）

```bash
# 1. 爬取中科大官网等数据源
python data/prepare_data.py

# 2. 用大模型 API 批量生成 QA 对
#    支持 OpenAI API / 通义千问 API 等兼容接口
export OPENAI_API_KEY="your-api-key"
# 如使用通义千问:
# export OPENAI_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"

python data/generate_qa.py --model gpt-4o-mini --num-pairs 10
```

生成的训练数据格式（Alpaca 格式）：
```json
[
  {
    "instruction": "中科大少年班是什么？",
    "input": "",
    "output": "中科大少年班创办于1978年..."
  }
]
```

### Step 2: LoRA 微调训练

#### 单卡训练

```bash
bash train/run_train.sh
```

#### 多卡分布式训练（推荐，2 × 5090）

```bash
# 训练脚本会自动检测 GPU 数量并启用 DeepSpeed ZeRO-2
bash train/run_train.sh
```

训练配置说明（`train/train_lora.yaml`）：

| 参数 | 值 | 说明 |
|------|-----|------|
| lora_rank | 64 | LoRA 秩，越大参数越多 |
| lora_alpha | 128 | LoRA 缩放系数 |
| learning_rate | 2e-4 | 学习率 |
| num_train_epochs | 3 | 训练轮数 |
| per_device_train_batch_size | 2 | 每卡 batch size |
| gradient_accumulation_steps | 8 | 梯度累积步数 |
| deepspeed | ZeRO-2 | 分布式训练策略 |

### Step 3: DPO 偏好对齐（推荐）

在 SFT 微调之后，使用 DPO（Direct Preference Optimization）进一步对齐模型，让模型学会生成更详细、更高质量的回答。

**DPO vs PPO (传统 RLHF)**：

| 方法 | 需要的模型数 | 复杂度 | 效果 |
|------|-------------|--------|------|
| PPO (RLHF) | 4 个 (Actor, Critic, Reward, Reference) | 高 | 好 |
| **DPO** | **2 个 (Policy, Reference)** | **低** | **接近 PPO** |

DPO 的核心思想：直接用偏好数据（chosen vs rejected）优化策略，不需要单独训练 Reward Model。

```bash
# 偏好数据格式: 每条包含 chosen (好回答) 和 rejected (差回答)
# 示例数据已在 data/preference_data.json 中

# 启动 DPO 训练 (需要先完成 Step 2 的 SFT)
bash train/run_dpo.sh
```

DPO 配置说明（`train/train_dpo.yaml`）：

| 参数 | 值 | 说明 |
|------|-----|------|
| stage | dpo | DPO 训练模式 |
| pref_beta | 0.1 | 偏好强度，越大越保守 |
| adapter_name_or_path | outputs/ustc-qa-lora | 在 SFT 模型基础上继续训练 |
| learning_rate | 5e-5 | DPO 学习率通常比 SFT 小 |
| num_train_epochs | 2 | DPO 通常训练轮数较少 |

### Step 4: veRL GRPO 强化学习（推荐）

在 DPO 之后，使用 **veRL GRPO** 进一步优化模型，通过规则奖励函数提升回答质量。

**GRPO vs PPO vs DPO**：

| 方法 | 需要的模型数 | 复杂度 | 效果 | 适用场景 |
|------|-------------|--------|------|----------|
| PPO (RLHF) | 4 个 (Actor, Critic, Reward, Reference) | 高 | 好 | 需要训练 Reward Model |
| DPO | 2 个 (Policy, Reference) | 低 | 接近 PPO | 有偏好数据 (chosen vs rejected) |
| **GRPO** | **1 个 (Policy)** | **最低** | **优秀** | **有规则奖励函数** |

**GRPO 核心原理**：
- **不需要 Critic 模型**：用组内相对排名估计优势函数
- **采样-对比-更新**：每个 prompt 采样 n 个回答，组内排序，计算相对优势
- **规则奖励**：直接用代码规则计算奖励（长度、格式、关键词、流畅度）

**veRL 框架简介**：
- 字节跳动开源的高效 RL 训练框架
- 支持 FSDP / Megatron 分布式训练
- 针对大模型 RL 优化，显存占用更低，训练速度更快

```bash
# 1. 准备 GRPO 训练数据（从现有 QA 数据生成）
python train/prepare_grpo_data.py

# 2. 启动 GRPO 训练（需要先完成 Step 3 的 DPO）
bash train/run_grpo.sh
```

GRPO 配置说明（`train/verl_config.yaml`）：

| 参数 | 值 | 说明 |
|------|-----|------|
| algorithm | grpo | GRPO 算法 |
| n | 4 | 每个 prompt 采样 4 个回答，组内相对排名 |
| clip_ratio | 0.2 | PPO 风格的梯度裁剪比例 |
| learning_rate | 1e-6 | RL 学习率通常比 DPO 更小 |
| batch_size | 16 | 训练批次大小 |
| max_epochs | 3 | GRPO 训练轮数 |
| reward_function | rule_based | 规则奖励函数 |

**奖励函数说明**（`train/reward_function.py`）：

| 奖励项 | 权重 | 说明 |
|--------|------|------|
| 长度奖励 | 0.2 | 回答长度在 50-300 字之间 |
| 格式奖励 | 0.3 | 包含分段、列表等格式 |
| 关键词奖励 | 0.3 | 包含问题中的关键信息 |
| 流畅度奖励 | 0.2 | 句子通顺，无重复 |

### Step 5: 合并 LoRA 权重

```bash
# 如果做了 DPO，使用 DPO 的 LoRA 权重
python train/merge_lora.py \
    --base-model Qwen/Qwen2.5-7B \
    --lora-adapter outputs/ustc-qa-dpo \
    --output outputs/ustc-qa-merged

# 如果只做了 SFT，使用 SFT 的 LoRA 权重
# python train/merge_lora.py \
#     --base-model Qwen/Qwen2.5-7B \
#     --lora-adapter outputs/ustc-qa-lora \
#     --output outputs/ustc-qa-merged
```

### Step 6: 模型量化

将合并后的模型转换为 GGUF Q8_0 格式（~7.5GB），适配 8GB 显存笔记本部署。

```bash
# GGUF Q8_0 量化（推荐，精度损失极小）
python quantize/quantize_model.py \
    --model-path outputs/ustc-qa-merged \
    --method gguf --bits 8

# 或 GPTQ INT4 量化（体积更小，~4.5GB）
python quantize/quantize_model.py \
    --model-path outputs/ustc-qa-merged \
    --method gptq --bits 4
```

### Step 7: llama.cpp 本地部署

使用 llama.cpp 在本地笔记本（RTX 5070 Laptop 8GB）上部署 GGUF 模型，提供 OpenAI 兼容 API。

```bash
# 安装 llama.cpp（macOS / Linux）
brew install llama.cpp
# 或从源码编译: https://github.com/ggerganov/llama.cpp

# 启动推理服务（OpenAI 兼容 API）
llama-server \
    -m outputs/ustc-qa-merged-gguf-q8_0/ustc-qa-Q8_0.gguf \
    --host 0.0.0.0 --port 8000 \
    -ngl 99 -c 4096

# 服务启动后，API 地址: http://localhost:8000/v1
```

测试 API：
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ustc-qa",
    "messages": [{"role": "user", "content": "中科大少年班是什么？"}],
    "max_tokens": 512
  }'
```

### Step 8: Web 聊天界面

```bash
# 启动 Gradio 界面（需要先启动 llama.cpp 服务）
python deploy/chat_demo.py --port 7860

# 浏览器访问: http://localhost:7860
```

### Step 9: 性能 Benchmark

```bash
# 运行性能测试（需要先启动 llama.cpp 服务）
python benchmark/benchmark.py --concurrency 4 --num-requests 10

# 结果保存至 benchmark/results.json
```

---

## 全链路流程图

```
┌─────────────────────────────────────────────────────────────────┐
│                    中科大智能问答助手 - 全链路                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │ 数据爬取  │───→│ GPT 生成 QA  │───→│ 数据清洗去重  │          │
│  └──────────┘    └──────────────┘    └──────┬───────┘          │
│                                             │                   │
│                                             ▼                   │
│  ┌──────────────────────────────────────────────────┐          │
│  │       SFT LoRA 微调 (DeepSpeed ZeRO-2)            │          │
│  │  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐         │          │
│  │  │ GPU0 │  │ GPU1 │  │ GPU2 │  │ GPU3 │         │          │
│  │  └──────┘  └──────┘  └──────┘  └──────┘         │          │
│  └──────────────────────┬───────────────────────────┘          │
│                         │                                       │
│                         ▼                                       │
│  ┌──────────────────────────────────────────────────┐          │
│  │         DPO 偏好对齐 (chosen vs rejected)         │          │
│  │  不需要 Reward Model，直接优化偏好                  │          │
│  └──────────────────────┬───────────────────────────┘          │
│                         │                                       │
│                         ▼                                       │
│  ┌──────────────────────────────────────────────────┐          │
│  │         veRL GRPO 强化学习 (规则奖励)             │          │
│  │  不需要 Critic，用组内相对排名估计优势              │          │
│  │  奖励函数：长度 + 格式 + 关键词 + 流畅度            │          │
│  └──────────────────────┬───────────────────────────┘          │
│                         │                                       │
│                         ▼                                       │
│  ┌──────────────┐    ┌───────────────┐                         │
│  │ LoRA 权重合并 │───→│ GGUF Q8_0 量化 │                         │
│  └──────────────┘    └──────┬────────┘                         │
│                             │                                   │
│                             ▼                                   │
│  ┌──────────────────────────────────────────────────┐          │
│  │           llama.cpp 推理服务                      │          │
│  │  GGUF 格式 + OpenAI 兼容 API                     │          │
│  │  GPU 加速 (ngl 99) + 4K Context                  │          │
│  └──────────────────────┬───────────────────────────┘          │
│                         │                                       │
│              ┌──────────┼──────────┐                           │
│              ▼          ▼          ▼                            │
│         ┌────────┐ ┌────────┐ ┌──────────┐                    │
│         │ Web UI │ │  API   │ │Benchmark │                    │
│         └────────┘ └────────┘ └──────────┘                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 关键技术原理

### DeepSpeed ZeRO-2

ZeRO（Zero Redundancy Optimizer）通过分片优化器状态和梯度来降低显存占用：

| 策略 | 分片内容 | 显存节省 |
|------|----------|----------|
| ZeRO-0 | 无分片 | 基准 |
| ZeRO-1 | 优化器状态 | ~4x |
| **ZeRO-2** | **优化器状态 + 梯度** | **~8x** |
| ZeRO-3 | 优化器状态 + 梯度 + 参数 | ~N倍 (N=GPU数) |

### PagedAttention (vLLM)

借鉴操作系统虚拟内存分页思想管理 KV Cache：
- 将 KV Cache 分为固定大小的 block
- 使用 block table 维护逻辑-物理映射
- 支持 copy-on-write，实现 beam search 等场景的内存共享
- 消除内存碎片，提升 GPU 显存利用率

### LoRA (Low-Rank Adaptation)

在预训练权重旁注入低秩矩阵，仅训练少量参数：
- 原始权重 W 冻结不变
- 新增 A (d×r) 和 B (r×d) 两个小矩阵，r << d
- 前向传播: h = Wx + BAx
- 参数量: 从数十亿降至数百万（~0.1%）

---

## 后续优化方向

以下优化可在基础版完成后逐步添加：

### 推理优化
- [ ] **Speculative Decoding**: 用小模型加速大模型推理 (1.5-2.5x)
- [ ] **FP8 推理**: 利用 RTX 5070 原生 FP8 支持
- [ ] **KV Cache 量化**: 将 KV Cache 压缩到 INT8/FP8
- [ ] **LoRA 热加载**: 一个基座服务多个 LoRA 适配器

### 训练优化
- [ ] **对比实验**: 单卡 vs 多卡加速比分析
- [ ] **ZeRO 对比**: ZeRO-0 vs ZeRO-2 vs ZeRO-3 显存/速度对比
- [ ] **超参搜索**: LoRA rank、学习率等参数的影响分析

### 数据优化
- [ ] **RAG 增强**: 结合向量数据库实现检索增强生成
- [ ] **数据增强**: 更多数据源、更多 QA 对
- [x] **DPO + GRPO**: 偏好对齐 + 规则奖励强化学习（已完成）

### 工程优化
- [ ] **量化对比**: GPTQ vs GGUF Q8_0 vs GGUF Q4_K_M 的精度/速度/显存对比
- [ ] **完整 Benchmark**: TTFT、吞吐量、并发性能的详细报告
- [ ] **Docker 部署**: 容器化部署方案

---

## 面试要点

本项目可以展示以下 AI Infra 核心能力：

| 能力维度 | 具体内容 |
|----------|----------|
| **数据工程** | 爬虫、数据清洗、GPT 生成 QA 对、Alpaca 格式 |
| **分布式训练** | DeepSpeed ZeRO-2、多卡并行、梯度累积、混合精度 |
| **参数高效微调** | LoRA 原理、rank/alpha 调参、权重合并 |
| **模型量化** | GPTQ/GGUF 原理、INT4/Q8_0 量化、精度-体积权衡 |
| **推理部署** | llama.cpp、GGUF 格式、OpenAI 兼容 API |
| **性能分析** | TTFT、吞吐量、加速比、瓶颈分析 |

---

## 参考资料

- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) - 一站式微调框架
- [vLLM](https://github.com/vllm-project/vllm) - 高性能推理引擎
- [DeepSpeed](https://github.com/microsoft/DeepSpeed) - 分布式训练框架
- [PEFT](https://github.com/huggingface/peft) - 参数高效微调库
- [Qwen2.5](https://github.com/QwenLM/Qwen2.5) - 基座模型

# 对比 SFT 模型
python3 train/evaluate_model.py --base-model /root/autodl-tmp/Qwen2.5-7B --lora-adapter outputs/ustc-qa-lora

# 对比 DPO 模型  
python3 train/evaluate_model.py --base-model outputs/ustc-qa-merged --lora-adapter outputs/ustc-qa-dpo

# 对比 GRPO 模型
python3 train/evaluate_model.py --base-model outputs/ustc-qa-grpo

## License

MIT License
