# ModelShield 项目

**本代码库及数据集对应论文: [ModelShield: 针对模型提取攻击的自适应鲁棒水印方案](https://arxiv.org/abs/2405.02365)**

---

## 项目概述

ModelShield 提供了一套完整的框架，用于在语言模型中生成、嵌入和验证水印，以防范模型提取攻击对知识产权的侵害。本代码库包含以下核心功能：
- 水印生成与验证系统
- 使用含水印数据训练模仿模型及生成模仿模型输出
- 实验支持数据集

---

## 环境依赖

环境配置仅在模型训练阶段需要。具体依赖项请参见[requirements文件](https://github.com/amaoku/ModelShield/blob/master/Imitation_Model_training/train/requirements.txt)。

---

## 实施流程

### 1. **水印生成阶段**
采用系统级指令引导语言模型生成水印，确保无缝集成与高鲁棒性（需不同LMaaS平台的API-KEY）

### 2. **模仿模型训练**
使用含水印数据微调模仿模型，模拟模型提取攻击场景

本模块基于[BELLE开源项目](https://github.com/LianjiaTech/BELLE)实现，主要特性包括：
- 支持**全参数微调**和**LoRA微调**
- 可灵活集成自定义微调方法
- 配置文件位于`config`目录（可设置基础模型、微调轮次、批大小、学习率及LoRA参数）

### 3. **水印验证阶段**
提供两种水印验证方法：
1. **快速验证**：基于文本快速检测水印存在
2. **详细验证**：需与原始模型及基础模型进行对比分析

---

## 实验数据集

本实验使用以下基准数据集：
- **HC3**：语言模型模仿分析专用数据集
- **WILD**：多场景鲁棒性评估数据集

---
## 使用指南
1. 从受害模型生成含水印数据（参见Watermark Generation目录说明）
2. 模拟模型提取攻击（参见Imitation Model training目录说明）
3. 执行水印验证（参见Watermark Verification目录说明）

---
## 引用文献

如果您的研究工作受益于本项目，请引用我们的论文：

```bibtex
@article{modelshield,
  title={Adaptive and robust watermark against model extraction attack},
  author={Pang, Kaiyi and Qi, Tao and Wu, Chuhan and Bai, Minhao and Jiang, Minghu and Huang, Yongfeng},
  journal={arXiv preprint arXiv:2405.02365},
  year={2024}
}
```