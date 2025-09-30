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
    在文本中查找水印词

    Args:
        text: 待搜索的文本
        watermark_words: 水印词列表

    Returns:
        List[str]: 找到的水印词列表
    """
    if not text or not isinstance(text, str):
        return []

    text_lower = text.lower()
    found_watermarks = []

    for word in watermark_words:
        if not word:
            continue

        word_lower = word.lower()

        # 使用正则表达式进行完整单词匹配
        pattern = r'\b' + re.escape(word_lower) + r'\b'
        if re.search(pattern, text_lower):
            found_watermarks.append(word)

    return found_watermarks

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

    total_watermarks_found = 0
    records_with_watermarks = 0

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

        # 获取prediction文本
        prediction = unwatermarked_data[i].get('prediction', '')

        # 在prediction中查找当前位置对应的水印词
        if position_watermark_words and prediction:
            found_watermarks = find_watermarks_in_text(prediction, position_watermark_words)
        else:
            found_watermarks = []

        # 更新watermark_words字段
        unwatermarked_data[i]['watermark_words'] = found_watermarks

        if found_watermarks:
            records_with_watermarks += 1
            total_watermarks_found += len(found_watermarks)

        if (i + 1) % 100 == 0:
            print(f"已处理 {i + 1} 条记录...")

    print("处理完成，正在保存文件...")

    # 保存更新后的文件
    try:
        with open(unwatermarked_file, 'w', encoding='utf-8') as f:
            json.dump(unwatermarked_data, f, ensure_ascii=False, indent=2)

        print(f"文件已保存: {unwatermarked_file}")

        # 输出统计结果
        print("\n=== 处理结果统计 ===")
        print(f"总记录数: {len(unwatermarked_data)}")
        print(f"实际处理记录数: {min_length}")
        print(f"包含水印词的记录数: {records_with_watermarks}")
        print(f"不包含水印词的记录数: {min_length - records_with_watermarks}")
        print(f"总共找到的水印词实例数: {total_watermarks_found}")
        print(f"平均每条记录的水印词数: {total_watermarks_found / min_length:.2f}")

    except Exception as e:
        print(f"保存文件时出错: {e}")
        raise

def main():
    """主函数"""
    # 文件路径
    watermarked_file = 'HC3_watermarked.json'
    unwatermarked_file = 'HC3_unwatermarked.json'

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