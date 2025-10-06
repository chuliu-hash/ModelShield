#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
处理human_answers的完整流程
1. 将all.json中的human_answers添加到目标JSON文件中
2. 将human_answers从列表转换为字符串格式
"""

import json
import os
from typing import Dict, List, Any, Optional

def load_json(file_path: str) -> List[Dict[str, Any]]:
    """加载JSON文件"""
    print(f"正在读取 {file_path}...")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"读取到 {len(data)} 条记录")
    return data

def save_json(file_path: str, data: List[Dict[str, Any]]) -> None:
    """保存JSON文件"""
    print(f"正在保存到 {file_path}...")

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"保存成功！")

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
    target_data: List[Dict[str, Any]],
    sentence_map: Dict[str, List[str]]
) -> List[Dict[str, Any]]:
    """
    将human_answers添加到目标数据中

    Args:
        target_data: 目标JSON数据
        sentence_map: sentence到human_answers的映射

    Returns:
        更新后的数据
    """
    matched_count = 0
    unmatched_count = 0

    for item in target_data:
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

    return target_data

def convert_human_answers_to_string(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    将human_answers从列表转换为字符串

    Args:
        data: JSON数据

    Returns:
        转换后的数据
    """
    converted_count = 0
    multi_answers_count = 0

    print("\n正在转换human_answers格式...")

    for i, item in enumerate(data):
        if 'human_answers' in item:
            human_answers = item['human_answers']

            # 如果是列表，转换为字符串
            if isinstance(human_answers, list):
                # 过滤掉 None 和空字符串
                human_answers = [ans for ans in human_answers if ans]

                if human_answers:
                    # 只有一个元素时，直接取出
                    if len(human_answers) == 1:
                        item['human_answers'] = human_answers[0]
                    else:
                        # 多个元素时，选择最长的一条
                        item['human_answers'] = max(human_answers, key=len)
                        multi_answers_count += 1
                else:
                    # 空列表转为空字符串
                    item['human_answers'] = ''

                converted_count += 1

        if (i + 1) % 50 == 0:
            print(f"  已处理 {i + 1} 条记录...")

    print(f"转换完成，共转换 {converted_count} 条记录")
    print(f"其中有多条human_answers的记录: {multi_answers_count} 条（已选择最长的一条）")

    return data

def create_backup(file_path: str) -> Optional[str]:
    """
    创建备份文件

    Args:
        file_path: 原文件路径

    Returns:
        备份文件路径，如果备份失败则返回None
    """
    if not os.path.exists(file_path):
        return None

    backup_file = f"{file_path}.backup"
    if not os.path.exists(backup_file):
        print(f"\n创建备份文件: {backup_file}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print("备份完成")
            return backup_file
        except Exception as e:
            print(f"备份失败: {e}")
            return None
    else:
        print(f"备份文件已存在: {backup_file}")
        return backup_file

def main():
    """主函数"""
    # 文件路径配置
    all_json_path = 'all.json'
    target_json_path = '../Watermark_Generation/input/HC3_watermarked_zh.json'
    output_json_path = '../Watermark_Generation/input/HC3_watermarked_zh_processed.json'

    # 是否跳过添加human_answers步骤（如果目标文件已包含human_answers）
    skip_add_step = False
    # 是否转换human_answers格式
    convert_to_string = True

    try:
        print("=== 开始处理human_answers ===\n")

        # 创建备份
        create_backup(target_json_path)

        # 加载目标文件
        target_data = load_json(target_json_path)

        # 步骤1: 添加human_answers（如果需要）
        if not skip_add_step:
            print("\n--- 步骤1: 添加human_answers ---")

            # 检查all.json是否存在
            if os.path.exists(all_json_path):
                # 加载all.json
                all_data = load_json(all_json_path)

                # 创建映射
                sentence_map = create_sentence_to_human_answer_map(all_data)

                # 添加human_answers
                target_data = add_human_answers(target_data, sentence_map)
            else:
                print(f"警告: {all_json_path} 不存在，跳过添加步骤")
                skip_add_step = True
        else:
            print("\n跳过添加human_answers步骤")

        # 步骤2: 转换human_answers格式（如果需要）
        if convert_to_string:
            print("\n--- 步骤2: 转换human_answers格式 ---")
            target_data = convert_human_answers_to_string(target_data)

        # 保存结果
        print("\n--- 保存结果 ---")
        save_json(output_json_path, target_data)

        # 验证结果
        print("\n--- 验证结果 ---")
        sample_indices = [0, min(9, len(target_data)-1), min(34, len(target_data)-1)]
        for idx in sample_indices:
            if idx < len(target_data) and 'human_answers' in target_data[idx]:
                print(f"\n记录 {idx}:")
                print(f"  sentence: {target_data[idx].get('sentence', '')[:50]}...")
                print(f"  human_answers 类型: {type(target_data[idx].get('human_answers', ''))}")
                print(f"  human_answers 前100字符: {str(target_data[idx].get('human_answers', ''))[:100]}")

        print("\n=== 处理完成 ===")
        print(f"结果已保存到: {output_json_path}")

    except Exception as e:
        print(f"\n程序执行出错: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
