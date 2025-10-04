#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将all.json中的human_answers添加到HC3_watermarked_zh.json中
根据sentence字段匹配，将human_answers插入到sentence下面
"""

import json
import os
from typing import Dict, List, Any

def load_json(file_path: str) -> List[Dict[str, Any]]:
    """加载JSON文件"""
    print(f"正在读取 {file_path}...")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"读取到 {len(data)} 条记录")
    return data

def create_sentence_to_human_answer_map(all_data: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    创建sentence到human_answers的映射

    Args:
        all_data: all.json的数据

    Returns:
        Dict: {sentence: human_answers} 的映射
    """
    sentence_map = {}

    for item in all_data:
        question = item.get('question', '').strip()
        human_answers = item.get('human_answers', [])

        if question:
            sentence_map[question] = human_answers

    print(f"创建了 {len(sentence_map)} 个sentence到human_answers的映射")
    return sentence_map

def add_human_answers(
    watermarked_data: List[Dict[str, Any]],
    sentence_map: Dict[str, List[str]]
) -> List[Dict[str, Any]]:
    """
    将human_answers添加到watermarked数据中

    Args:
        watermarked_data: HC3_watermarked_zh.json的数据
        sentence_map: sentence到human_answers的映射

    Returns:
        更新后的数据
    """
    matched_count = 0
    unmatched_count = 0

    for item in watermarked_data:
        sentence = item.get('sentence', '').strip()

        if sentence in sentence_map:
            # 找到匹配，添加human_answers
            item['human_answers'] = sentence_map[sentence]
            matched_count += 1
        else:
            # 未找到匹配
            unmatched_count += 1

    print(f"\n匹配统计:")
    print(f"  成功匹配: {matched_count} 条")
    print(f"  未找到匹配: {unmatched_count} 条")

    return watermarked_data

def save_json(file_path: str, data: List[Dict[str, Any]]) -> None:
    """保存JSON文件"""
    print(f"\n正在保存到 {file_path}...")

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"保存成功！")

def main():
    """主函数"""
    # 文件路径
    all_json_path = 'all.json'
    watermarked_json_path = '../Watermark_Generation/input/HC3_watermarked_zh.json'
    output_json_path = '../Watermark_Generation/input/HC3_watermarked_zh.json'  # 覆盖原文件，如需保留原文件可改名

    try:
        print("=== 开始处理 ===\n")

        # 1. 加载all.json
        all_data = load_json(all_json_path)

        # 2. 加载HC3_watermarked_zh.json
        watermarked_data = load_json(watermarked_json_path)

        # 3. 创建sentence到human_answers的映射
        sentence_map = create_sentence_to_human_answer_map(all_data)

        # 4. 添加human_answers到watermarked数据
        updated_data = add_human_answers(watermarked_data, sentence_map)

        # 5. 保存结果
        save_json(output_json_path, updated_data)

        print("\n=== 处理完成 ===")

    except Exception as e:
        print(f"\n程序执行出错: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
