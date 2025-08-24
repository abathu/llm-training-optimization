# llm-training-optimization



# Qwen/Qwen2.5-7B-Instruct 模型训练与优化 (Training & Inference Optimization)

目标是满足实际业务中 **高效训练 + 快速推理 + 部署落地** 的需求。

## 项目目标

- 在现有Qwen2.5-7B   从零训练基础上，加入**混合精度、梯度优化、分布式加速**等训练优化方法；
- 研究 **推理加速**（TensorRT、ONNX、量化）在NLP场景中的可行性；
- 输出可量化的优化效果报告（显存占用、训练速度、推理延迟对比）。

## 已完成
- [x] **训练优化 (Qwen2.5-7B + PEFT/QLoRA)**  
  - 混合精度训练（FP16/BF16，`torch.cuda.amp` / DeepSpeed AMP）  
  - 梯度累积 & 梯度检查点（减少显存占用）  
  - DeepSpeed ZeRO-3 多卡分布式训练验证  
- [x] **PEFT LoRA 部分实现**（Q/K/V/O/MLP 投影层 LoRA 注入）  
- [x] **多 GPU 正常运行**（A6000 双卡实验验证）

## 显存优化对比

| 训练方式                     | 显存占用 (2×A6000) | 显存占用 (单卡) | 减少量 | 减少百分比 |
|------------------------------|-------------------|----------------|--------|------------|
| 全参数微调 (FP16)            | ≈ 120 GB          | ≈ 60 GB        | –      | –          |
| QLoRA (4bit + LoRA Adapter)  | ≈ 36 GB           | ≈ 18 GB        | 84 GB  | **≈ 70% ↓** |

---

## 改进计划（按优先级排序）

### 1. 训练优化（Highest Priority）  Done
- 混合精度训练（FP16/BF16）  
- 梯度累积与梯度检查点  
- DeepSpeed ZeRO/FSDP 优化  

**预期成果**  
-  训练速度提升 **1.5~2 倍**  
   显存占用降低 **30~50%**

---

### 2. 推理优化 ⬜Todo
- 模型量化（FP16/INT8/Q4）加速推理  
- TensorRT / ONNX Runtime 部署推理  
- 批量推理 + KV Cache 优化  

**预期成果**  
- 推理延迟降低 **40~60%**  
- 吞吐量提升 **2 倍以上**

---

### 3. 分布式与架构扩展 ⬜ Todo
- 多节点分布式训练（SLURM + `torchrun`）  
- Pipeline Parallel / Tensor Parallel 探索  

**预期成果**  
- 展示多节点训练加速比  
- 支持更大模型规模训练  

---

### 4. 业务场景结合 ⬜ Todo
- 使用评论/标题语料微调  
- 构建 **RAG + GPT-2 电商问答 Demo**  

**预期成果**  
- 📉 PPL 降低  
- 🎯 问答准确率提升  

---

### 5. 工程化与展示 ⬜ Todo
- Docker + Triton 容器化部署  
- TensorBoard/W&B 记录训练与推理性能  
- README 增加性能对比表 + Demo 链接  

---

## 性能对比（示例占位）

| 优化策略        | 显存占用 (GB) | 训练速度 (tokens/s) | 推理延迟 (ms/token) |
|-----------------|---------------|---------------------|---------------------|
| 原始 FP32       | 24            | 1000                | 50                  |
| AMP (FP16)      | 12            | 1800                | 30                  |
| ZeRO + FP16     | 8             | 2200                | 28                  |
| TensorRT (INT8) | -             | -                   | 15                  |

---

## 技术栈

- **模型训练**: PyTorch, Hugging Face Transformers, PyTorch Lightning
- **优化框架**: DeepSpeed, FSDP, AMP
- **推理加速**: TensorRT, ONNX Runtime, bitsandbytes
- **部署工具**: Docker, Triton Inference Server, SLURM
- **业务扩展**: FAISS, RAG, 语料微调

---

## 目录结构（建议）
