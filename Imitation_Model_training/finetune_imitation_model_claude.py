#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLaMA2-7B Fine-tuning Script with LoRA Support
支持中文微调、检查点恢复、LoRA参数高效微调
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field

import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    set_seed,
)
from datasets import Dataset
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)

# 尝试导入prepare_model_for_kbit_training（旧版本peft需要）
try:
    from peft import prepare_model_for_kbit_training
except ImportError:
    # 新版本peft已移除此函数，提供一个兼容的替代
    def prepare_model_for_kbit_training(model, use_gradient_checkpointing=True):
        """
        兼容函数：为量化训练准备模型
        在新版本的peft中，这个函数已被移除，模型准备逻辑已内置
        """
        # 启用梯度检查点以节省显存
        if use_gradient_checkpointing:
            model.gradient_checkpointing_enable()

        # 确保输入嵌入层可训练
        for param in model.parameters():
            param.requires_grad = False

        # 启用输入需要梯度（用于梯度检查点）
        model.enable_input_require_grads()

        return model

# 配置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("training.log", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """模型参数"""
    model_name_or_path: str = field(metadata={"help": "预训练模型路径"})
    use_lora: bool = field(default=False, metadata={"help": "是否使用LoRA"})
    lora_r: int = field(default=16, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout"})
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"],
        metadata={"help": "LoRA目标模块"}
    )


@dataclass
class DataArguments:
    """数据参数"""
    data_path: str = field(metadata={"help": "训练数据路径"})
    cutoff_len: int = field(default=512, metadata={"help": "最大序列长度"})
    val_set_size: Optional[int] = field(default=None, metadata={"help": "验证集大小"})
    val_set_rate: float = field(default=0.1, metadata={"help": "验证集比例"})


def load_config(config_file: str) -> Dict:
    """加载配置文件"""
    with open(config_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_training_data(data_path: str, tokenizer, cutoff_len: int, val_set_size: Optional[int] = None,
                       val_set_rate: float = 0.1, model_type: str = "base"):
    """
    加载并处理训练数据

    Args:
        data_path: 数据文件路径
        tokenizer: 分词器
        cutoff_len: 最大序列长度
        val_set_size: 验证集大小
        val_set_rate: 验证集比例
        model_type: 模型类型 ("base" 或 "chat")

    Returns:
        train_dataset, val_dataset
    """
    logger.info(f"Loading data from {data_path}")

    # 读取数据
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    logger.info(f"Loaded {len(data)} samples")
    logger.info(f"Using model type: {model_type}")

    # 数据格式转换
    def format_sample(sample: Dict) -> str:
        """格式化单个样本为训练格式"""
        instruction = sample.get('sentence', '').strip()
        output = sample.get('prediction', '').strip()

        if model_type == "chat":
            # LLaMA2-Chat格式（带指令标记）
            text = f"<s>[INST] {instruction} [/INST] {output} </s>"
        else:
            # LLaMA2-Base格式（简单拼接，适用于基础模型）
            # 使用更自然的问答格式
            text = f"### 问题: {instruction}\n\n### 回答: {output}"

        return text

    def tokenize_function(sample):
        """分词函数 - 处理单个样本"""
        # 格式化文本
        text = format_sample(sample)

        # 分词
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )

        # 设置labels（用于计算loss）
        tokenized["labels"] = tokenized["input_ids"].copy()

        return tokenized

    # 转换为Dataset格式
    dataset = Dataset.from_list(data)

    # 分割训练集和验证集
    if val_set_size is None:
        val_set_size = int(len(dataset) * val_set_rate)

    if val_set_size > 0:
        dataset_split = dataset.train_test_split(test_size=val_set_size, seed=42)
        train_dataset = dataset_split['train']
        val_dataset = dataset_split['test']
    else:
        train_dataset = dataset
        val_dataset = None

    # 应用分词
    logger.info("Tokenizing training data...")
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=False,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train dataset",
    )

    if val_dataset is not None:
        logger.info("Tokenizing validation data...")
        val_dataset = val_dataset.map(
            tokenize_function,
            batched=False,
            remove_columns=val_dataset.column_names,
            desc="Tokenizing val dataset",
        )

    logger.info(f"Train samples: {len(train_dataset)}")
    if val_dataset:
        logger.info(f"Val samples: {len(val_dataset)}")

    return train_dataset, val_dataset


def setup_model_and_tokenizer(model_args: ModelArguments, training_args: TrainingArguments):
    """
    设置模型和分词器

    Args:
        model_args: 模型参数
        training_args: 训练参数

    Returns:
        model, tokenizer
    """
    logger.info(f"Loading model from {model_args.model_name_or_path}")

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        use_fast=False,
    )

    # 设置特殊token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    tokenizer.padding_side = "left"  # LLaMA使用左padding

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # 设置模型配置
    model.config.use_cache = False  # 训练时禁用cache

    if model_args.use_lora:
        logger.info("Applying LoRA configuration...")

        # 准备模型（如果使用量化）
        model = prepare_model_for_kbit_training(model)

        # LoRA配置
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=model_args.lora_target_modules,
            lora_dropout=model_args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        # 应用LoRA
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        logger.info("Using full fine-tuning")
        # 全量微调：启用梯度
        for param in model.parameters():
            param.requires_grad = True

    return model, tokenizer


def train(
        model_config_file: str,
        lora_hyperparams_file: Optional[str] = None,
        use_lora: bool = False,
        resume_from_checkpoint: bool = False,
        deepspeed: Optional[str] = None,
):
    """
    主训练函数

    Args:
        model_config_file: 模型配置文件路径
        lora_hyperparams_file: LoRA配置文件路径
        use_lora: 是否使用LoRA
        resume_from_checkpoint: 是否从检查点恢复
        deepspeed: DeepSpeed配置文件路径
    """
    # 加载配置
    config = load_config(model_config_file)
    logger.info(f"Model config: {json.dumps(config, indent=2, ensure_ascii=False)}")

    # 加载LoRA配置
    if use_lora and lora_hyperparams_file:
        lora_config = load_config(lora_hyperparams_file)
        logger.info(f"LoRA config: {json.dumps(lora_config, indent=2, ensure_ascii=False)}")
    else:
        lora_config = {}

    # 设置随机种子
    set_seed(42)

    # 准备模型参数
    model_args = ModelArguments(
        model_name_or_path=config["model_name_or_path"],
        use_lora=use_lora,
        lora_r=lora_config.get("lora_r", 16),
        lora_alpha=lora_config.get("lora_alpha", 32),
        lora_dropout=lora_config.get("lora_dropout", 0.05),
        lora_target_modules=lora_config.get("lora_target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
    )

    # 准备训练参数
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        num_train_epochs=config.get("num_epochs", 3),
        per_device_train_batch_size=config.get("per_device_train_batch_size", 1),
        per_device_eval_batch_size=config.get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 16),
        learning_rate=float(config.get("learning_rate", 2e-5)),
        warmup_steps=config.get("warmup_steps", 10),
        logging_steps=config.get("logging_steps", 10),
        save_steps=config.get("save_steps", 500),
        eval_steps=config.get("eval_steps", 500),
        save_total_limit=3,
        fp16=True,
        evaluation_strategy="steps" if config.get("val_set_size", 0) > 0 else "no",
        save_strategy="steps",
        load_best_model_at_end=True if config.get("val_set_size", 0) > 0 else False,
        report_to="tensorboard",
        remove_unused_columns=False,
        ddp_find_unused_parameters=False if deepspeed else None,
        deepspeed=deepspeed,
        gradient_checkpointing=True,
        resume_from_checkpoint=resume_from_checkpoint,
    )

    # 设置模型和分词器
    model, tokenizer = setup_model_and_tokenizer(model_args, training_args)

    # 确定模型类型（base或chat）
    is_chat_model = config.get("is_chat_model", False)
    model_format = "chat" if is_chat_model else "base"
    logger.info(f"Model format: {model_format} (is_chat_model={is_chat_model})")

    # 加载数据
    train_dataset, val_dataset = load_training_data(
        data_path=config["data_path"],
        tokenizer=tokenizer,
        cutoff_len=config.get("cutoff_len", 512),
        val_set_size=config.get("val_set_size"),
        val_set_rate=config.get("val_set_rate", 0.1),
        model_type=model_format,
    )

    # 数据整理器
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )

    # 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    # 检查是否有检查点
    checkpoint = None
    if resume_from_checkpoint:
        checkpoint_dir = Path(config["output_dir"])
        if checkpoint_dir.exists():
            checkpoints = list(checkpoint_dir.glob("checkpoint-*"))
            if checkpoints:
                # 获取最新的检查点
                checkpoint = max(checkpoints, key=lambda x: int(x.name.split("-")[-1]))
                logger.info(f"Found checkpoint: {checkpoint}")

    # 开始训练
    logger.info("Starting training...")
    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    # 保存模型
    logger.info(f"Saving model to {config['output_dir']}")

    if use_lora:
        # LoRA模式：只保存适配器权重，节省存储空间
        logger.info("Saving LoRA adapters only (not saving full model to save space)...")
        model.save_pretrained(config["output_dir"])
        tokenizer.save_pretrained(config["output_dir"])

        logger.info(f"LoRA adapters saved to {config['output_dir']}")
        logger.info(f"Adapter size is much smaller than full model (~50-200MB vs ~13GB)")

    else:
        # 全量微调：保存完整模型
        logger.info("Saving full fine-tuned model...")
        trainer.save_model(config["output_dir"])
        tokenizer.save_pretrained(config["output_dir"])
        logger.info(f"Full model saved to {config['output_dir']}")

    # 保存训练指标
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("Training completed!")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Fine-tune LLaMA2-7B with LoRA support")
    parser.add_argument(
        "--model_config_file",
        type=str,
        default="./config/llama2.json",
        help="模型配置文件路径"
    )
    parser.add_argument(
        "--lora_hyperparams_file",
        type=str,
        default="./config/lora_config_llama.json",
        help="LoRA配置文件路径"
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        default=True,
        help="是否使用LoRA"
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        action="store_true",
        default=False,
        help="是否从检查点恢复训练"
    )
    parser.add_argument(
        "--deepspeed",
        type=str,
        default=None,
        help="DeepSpeed配置文件路径"
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="分布式训练的本地rank"
    )

    args = parser.parse_args()

    # 训练
    train(
        model_config_file=args.model_config_file,
        lora_hyperparams_file=args.lora_hyperparams_file,
        use_lora=args.use_lora,
        resume_from_checkpoint=args.resume_from_checkpoint,
        deepspeed=args.deepspeed,
    )


if __name__ == "__main__":
    main()