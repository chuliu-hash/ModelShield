#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
水印词处理脚本
功能：
1. 从HC3_watermarked.json中收集所有唯一的水印词
2. 遍历HC3_unwatermarked.json中的prediction字段
3. 查找匹配的水印词并写入watermark_words字段
"""

import json
import re
import os
from typing import List, Set, Dict, Any

def collect_unique_watermark_words(watermarked_file: str) -> List[str]:
    """
    从HC3_watermarked.json中收集所有唯一的水印词

    Args:
        watermarked_file: HC3_watermarked.json文件路径

    Returns:
        List[str]: 唯一水印词列表
    """
    print(f"正在读取 {watermarked_file}...")

    if not os.path.exists(watermarked_file):
        raise FileNotFoundError(f"文件不存在: {watermarked_file}")

    watermark_words_set: Set[str] = set()

    try:
        with open(watermarked_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"读取到 {len(data)} 条记录")

        for i, item in enumerate(data):
            if isinstance(item, dict) and 'watermark_words' in item:
                watermark_words = item['watermark_words']
                if watermark_words and isinstance(watermark_words, list):
                    for word in watermark_words:
                        if word and isinstance(word, str):
                            watermark_words_set.add(word.strip())

            if (i + 1) % 1000 == 0:
                print(f"已处理 {i + 1} 条记录...")

    except json.JSONDecodeError as e:
        print(f"JSON格式错误: {e}")

    except Exception as e:
        print(f"读取文件时出错: {e}")
        raise

    unique_words = list(watermark_words_set)
    print(f"收集到 {len(unique_words)} 个唯一的水印词")

    return unique_words

def find_watermarks_in_text(text: str, watermark_words: List[str]) -> List[str]:
    """
    在文本中查找水印词（支持中文）

    Args:
        text: 待搜索的文本
        watermark_words: 水印词列表

    Returns:
        List[str]: 找到的水印词列表
    """
    if not text or not isinstance(text, str):
        return []

    found_watermarks = []

    for word in watermark_words:
        if not word:
            continue

        # 直接使用in操作符或re.search进行子串匹配
        # 对于中文文本，直接查找子串即可
        if word in text:
            found_watermarks.append(word)

    return found_watermarks

def calculate_text_length(text: str) -> int:
    """
    计算文本长度（字符数）

    Args:
        text: 输入文本

    Returns:
        int: 文本字符数
    """
    return len(text) if text else 0

def calculate_watermark_length(watermark_words: List[str]) -> int:
    """
    计算水印词总长度

    Args:
        watermark_words: 水印词列表

    Returns:
        int: 水印词总字符数
    """
    return sum(len(word) for word in watermark_words if word)

def process_files_by_position(watermarked_file: str, unwatermarked_file: str) -> None:
    """
    按位置匹配处理HC3文件，使用相同位置的水印词进行查找

    Args:
        watermarked_file: HC3_watermarked.json文件路径
        unwatermarked_file: HC3_unwatermarked.json文件路径
    """
    print(f"正在读取 {watermarked_file}...")

    # 读取watermarked文件
    if not os.path.exists(watermarked_file):
        raise FileNotFoundError(f"文件不存在: {watermarked_file}")

    try:
        with open(watermarked_file, 'r', encoding='utf-8') as f:
            watermarked_data = json.load(f)
        print(f"读取到watermarked数据 {len(watermarked_data)} 条记录")

    except json.JSONDecodeError as e:
        print(f"Watermarked文件JSON格式错误: {e}")
        raise

    print(f"正在读取 {unwatermarked_file}...")

    # 读取unwatermarked文件
    if not os.path.exists(unwatermarked_file):
        raise FileNotFoundError(f"文件不存在: {unwatermarked_file}")

    try:
        with open(unwatermarked_file, 'r', encoding='utf-8') as f:
            unwatermarked_data = json.load(f)

        print(f"读取到unwatermarked数据 {len(unwatermarked_data)} 条记录")

    except json.JSONDecodeError as e:
        print(f"Unwatermarked文件JSON格式错误: {e}")
        raise

    # 确保两个文件的记录数量一致
    min_length = min(len(watermarked_data), len(unwatermarked_data))
    print(f"将处理 {min_length} 条记录（取两个文件的最小长度）")

    # Prediction统计
    total_watermarks_found = 0
    records_with_watermarks = 0
    total_watermark_chars = 0
    total_text_chars = 0

    # Human answers统计
    human_total_watermarks_found = 0
    human_records_with_watermarks = 0
    human_total_watermark_chars = 0
    human_total_text_chars = 0

    for i in range(min_length):
        if not isinstance(unwatermarked_data[i], dict):
            unwatermarked_data[i] = {}

        # 获取当前位置的水印词
        if (i < len(watermarked_data) and
            isinstance(watermarked_data[i], dict) and
            'watermark_words' in watermarked_data[i]):

            position_watermark_words = watermarked_data[i]['watermark_words']

            # 确保watermark_words是列表
            if not isinstance(position_watermark_words, list):
                position_watermark_words = []
        else:
            position_watermark_words = []

        # === 处理 Prediction ===
        prediction = unwatermarked_data[i].get('prediction', '')
        text_length = calculate_text_length(prediction)
        total_text_chars += text_length

        # 在prediction中查找水印词
        if position_watermark_words and prediction:
            found_watermarks = find_watermarks_in_text(prediction, position_watermark_words)
        else:
            found_watermarks = []

        watermark_length = calculate_watermark_length(found_watermarks)
        total_watermark_chars += watermark_length

        if found_watermarks:
            records_with_watermarks += 1
            total_watermarks_found += len(found_watermarks)

        # === 处理 Human Answers ===
        human_answers = unwatermarked_data[i].get('human_answers', [])

        if human_answers and isinstance(human_answers, list):
            # 将所有human_answers合并成一个文本
            human_text = ' '.join(human_answers)
            human_text_length = calculate_text_length(human_text)
            human_total_text_chars += human_text_length

            # 在human_answers中查找水印词
            if position_watermark_words:
                human_found_watermarks = find_watermarks_in_text(human_text, position_watermark_words)
            else:
                human_found_watermarks = []

            human_watermark_length = calculate_watermark_length(human_found_watermarks)
            human_total_watermark_chars += human_watermark_length

            if human_found_watermarks:
                human_records_with_watermarks += 1
                human_total_watermarks_found += len(human_found_watermarks)

        if (i + 1) % 100 == 0:
            print(f"已处理 {i + 1} 条记录...")

    print("处理完成...")

    # 计算平均水印密度
    avg_watermark_density = (total_watermark_chars / total_text_chars * 100) if total_text_chars > 0 else 0.0
    human_avg_watermark_density = (human_total_watermark_chars / human_total_text_chars * 100) if human_total_text_chars > 0 else 0.0

    # 输出统计结果
    print("\n" + "="*70)
    print("=== Prediction (模型输出) 水印统计 ===")
    print("="*70)
    print(f"总记录数: {min_length}")
    print(f"包含水印词的记录数: {records_with_watermarks}")
    print(f"不包含水印词的记录数: {min_length - records_with_watermarks}")
    print(f"水印词覆盖率: {records_with_watermarks / min_length * 100:.2f}%")
    print(f"总共找到的水印词实例数: {total_watermarks_found}")
    print(f"平均每条记录的水印词数: {total_watermarks_found / min_length:.2f}")
    print(f"文本总字符数: {total_text_chars:,}")
    print(f"水印词总字符数: {total_watermark_chars:,}")
    print(f"平均水印密度: {avg_watermark_density:.4f}%")

    print("\n" + "="*70)
    print("=== Human Answers (人类答案) 水印统计 ===")
    print("="*70)
    print(f"总记录数: {min_length}")
    print(f"包含水印词的记录数: {human_records_with_watermarks}")
    print(f"不包含水印词的记录数: {min_length - human_records_with_watermarks}")
    print(f"水印词覆盖率: {human_records_with_watermarks / min_length * 100:.2f}%")
    print(f"总共找到的水印词实例数: {human_total_watermarks_found}")
    print(f"平均每条记录的水印词数: {human_total_watermarks_found / min_length:.2f}")
    print(f"文本总字符数: {human_total_text_chars:,}")
    print(f"水印词总字符数: {human_total_watermark_chars:,}")
    print(f"平均水印密度: {human_avg_watermark_density:.4f}%")

    print("\n" + "="*70)
    print("=== 对比分析 ===")
    print("="*70)
    print(f"Prediction vs Human - 覆盖率差异: {(records_with_watermarks - human_records_with_watermarks) / min_length * 100:+.2f}%")
    print(f"Prediction vs Human - 水印词数差异: {(total_watermarks_found - human_total_watermarks_found) / min_length:+.2f} 个/记录")
    print(f"Prediction vs Human - 水印密度差异: {avg_watermark_density - human_avg_watermark_density:+.4f}%")
    print("="*70)

def main():
    """主函数"""
    # 文件路径
    watermarked_file = '../Watermark_Generation/input/HC3_watermarked_zh.json'
    unwatermarked_file = '../Watermark Verification/data/finetuned_output.json'

    try:
        print("=== 开始按位置处理水印词 ===")

        # 使用新的按位置匹配方法
        process_files_by_position(watermarked_file, unwatermarked_file)

        print("\n=== 处理完成 ===")

    except Exception as e:
        print(f"\n程序执行出错: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())