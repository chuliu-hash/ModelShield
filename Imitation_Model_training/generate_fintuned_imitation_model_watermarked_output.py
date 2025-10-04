#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速开始示例 - 展示如何使用训练好的模型进行推理
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json
import argparse


def load_lora_model(base_model_path: str, lora_path: str):
    """
    加载LoRA微调模型

    Args:
        base_model_path: 基础模型路径
        lora_path: LoRA适配器路径（训练时的 output_dir）
    """
    print(f"加载基础模型: {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        load_in_8bit=False,  # 可选：启用8位量化以节省显存，但可能略微降低速度
    )

    print(f"加载LoRA适配器: {lora_path}")
    model = PeftModel.from_pretrained(model, lora_path)
    model.eval()

    return model, tokenizer


def load_full_model(model_path: str):
    """加载全量微调模型"""
    print(f"加载模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        load_in_8bit=False,  # 可选：启用8位量化以节省显存，但可能略微降低速度
    )
    model.eval()

    return model, tokenizer


def generate_response(
        model,
        tokenizer,
        prompt: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        model_type: str = "base"
):
    """生成回答"""
    # 根据模型类型格式化输入
    if model_type == "chat":
        # LLaMA2-Chat格式
        formatted_prompt = f"<s>[INST] {prompt} [/INST] "
    else:
        # LLaMA2-Base格式
        formatted_prompt = f"### 问题: {prompt}\n\n### 回答:"

    # 编码
    inputs = tokenizer(formatted_prompt, return_tensors="pt")

    # 过滤掉 LLaMA 不需要的参数（如 token_type_ids）
    inputs = {
        k: v.to(model.device)
        for k, v in inputs.items()
        if k in ['input_ids', 'attention_mask']
    }

    # 生成（使用inference_mode获得更好的性能）
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,  # 启用KV cache加速
            num_beams=1,  # 贪婪解码，避免beam search开销
        )

    # 解码
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 提取回答部分
    if model_type == "chat" and "[/INST]" in full_text:
        response = full_text.split("[/INST]")[-1].strip()
    elif model_type == "base" and "### 回答:" in full_text:
        response = full_text.split("### 回答:")[-1].strip()
    else:
        response = full_text

    return response


def interactive_mode(model, tokenizer, model_type="base"):
    """交互式问答"""
    print("\n" + "=" * 60)
    print("交互式问答模式")
    print(f"模型格式: {model_type}")
    print("输入 'exit' 或 'quit' 退出")
    print("=" * 60 + "\n")

    while True:
        try:
            question = input("\n问题: ").strip()

            if question.lower() in ['exit', 'quit', '退出']:
                print("再见！")
                break

            if not question:
                continue

            print("生成中...")
            response = generate_response(model, tokenizer, question, model_type=model_type)
            print(f"\n回答: {response}")

        except KeyboardInterrupt:
            print("\n\n再见！")
            break
        except Exception as e:
            print(f"\n错误: {e}")


def batch_mode(model, tokenizer, input_file: str, output_file: str, model_type="base", batch_size: int = 1):
    """批量处理模式 - 支持实时输出和断点恢复"""
    print(f"\n批量处理模式")
    print(f"模型格式: {model_type}")
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print(f"批处理大小: {batch_size}")

    # 读取输入数据
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total = len(data)
    print(f"总样本数: {total}")

    # 检查是否存在未完成的输出文件（断点恢复）
    results = []
    start_idx = 0

    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            start_idx = len(results)
            print(f"检测到已处理 {start_idx} 个样本，从第 {start_idx + 1} 个继续...")
        except (json.JSONDecodeError, Exception) as e:
            print(f"无法读取已有结果文件: {e}")
            print("将从头开始处理")
            results = []
            start_idx = 0

    # 从断点处继续处理（批量处理优化）
    for batch_start in range(start_idx, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_items = data[batch_start:batch_end]

        for idx, item in enumerate(batch_items):
            i = batch_start + idx
            question = item.get('sentence', '')

            # 显示进度和问题
            print(f"\n[{i + 1}/{total}] 处理: {question[:80]}{'...' if len(question) > 80 else ''}")

            try:
                response = generate_response(model, tokenizer, question, model_type=model_type)

                # 实时输出生成的回答
                print(f"✓ 回答: {response[:100]}{'...' if len(response) > 100 else ''}")

                result = {
                    'sentence': question,
                    'prediction': response,
                    'human_answers': item['human_answers'],
                    'watermark_words': item['watermark_words']
                }
                results.append(result)

            except Exception as e:
                print(f"✗ 错误: {e}")
                result = {
                    'question': question,
                    'response': f"ERROR: {str(e)}",
                    'human_answers': item['human_answers'],
                    'watermark_words': item['watermark_words']

                }
                results.append(result)

        # 每个batch处理完后统一保存一次（减少I/O开销）
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print(f"✓ 全部完成！共处理 {len(results)} 个样本")
    print(f"✓ 结果已保存到: {output_file}")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="LLaMA2 微调模型推理")
    parser.add_argument(
        "--mode",
        type=str,
        choices=['lora', 'full'],
        default='full',
        help="模型类型"
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="./Llama-2-7b-hf/",
        help="基础模型路径（LoRA模式需要）"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./finetuned_merged/",
        help="微调模型路径"
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="LoRA适配器路径（如果与model_path不同）"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="交互式问答模式"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        default="True",
        help="批量处理模式"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="./data/HC3_watermarked_zh.json",
        help="批量处理输入文件"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="finetuned_output.json",
        help="批量处理输出文件"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=['base', 'chat'],
        default='base',
        help="模型格式类型：base（基础模型）或 chat（对话模型）"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="批处理保存间隔（每处理N个样本保存一次，减少I/O开销）"
    )

    args = parser.parse_args()

    # 加载模型
    print("\n" + "=" * 60)
    print("加载模型")
    print("=" * 60 + "\n")

    if args.mode == 'lora':
        # LoRA 模式：适配器直接在 output_dir 中，不再在子目录
        lora_path = args.lora_path or args.model_path
        print(f"LoRA 适配器路径: {lora_path}")
        print(f"基础模型路径: {args.base_model_path}")
        model, tokenizer = load_lora_model(args.base_model_path, lora_path)
    else:
        model, tokenizer = load_full_model(args.model_path)

    print("\n模型加载完成！\n")
    print(f"使用模型格式: {args.model_type}\n")

    # 运行模式
    if args.batch and args.input_file and args.output_file:
        batch_mode(model, tokenizer, args.input_file, args.output_file, model_type=args.model_type, batch_size=args.batch_size)
    elif args.interactive or (not args.batch):
        interactive_mode(model, tokenizer, model_type=args.model_type)
    else:
        print("错误: 批量模式需要指定 --input_file 和 --output_file")


if __name__ == "__main__":
    main()