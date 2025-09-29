#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JSONL to JSON Converter Script
将.jsonl文件转换为.json文件的Python脚本
"""

import json
import os
from typing import List, Dict, Any


def convert_jsonl_to_json(input_file: str, output_file: str = None) -> None:
    """
    将JSONL文件转换为JSON文件

    Args:
        input_file (str): 输入的JSONL文件路径
        output_file (str): 输出的JSON文件路径，如果为None则自动生成
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"输入文件不存在: {input_file}")

    if output_file is None:
        # 自动生成输出文件名：将.jsonl扩展名替换为.json
        output_file = input_file.rsplit('.jsonl', 1)[0] + '.json'

    data_list: List[Dict[str, Any]] = []

    try:
        # 读取JSONL文件
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:  # 跳过空行
                    try:
                        json_obj = json.loads(line)

                        # 删除chatgpt_answers字段（如果存在）
                        if 'chatgpt_answers' in json_obj:
                            del json_obj['chatgpt_answers']

                        # 创建新的有序字典，id字段排在第一位
                        ordered_obj = {'id': line_num - 1}  # 从0开始编号
                        ordered_obj.update(json_obj)

                        data_list.append(ordered_obj)
                    except json.JSONDecodeError as e:
                        print(f"警告：第{line_num}行JSON解析错误: {e}")
                        continue

        # 写入JSON文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data_list, f, ensure_ascii=False, indent=2)

        print(f"转换完成！")
        print(f"输入文件: {input_file}")
        print(f"输出文件: {output_file}")
        print(f"共转换 {len(data_list)} 条记录")

    except Exception as e:
        print(f"转换过程中发生错误: {e}")
        raise


def main():
    """主函数"""
    # 直接在代码中指定输入和输出文件路径
    input_file = "all.jsonl"  # 输入文件路径
    output_file = "all.json"  # 输出文件路径，如果设为None则自动生成

    try:
        convert_jsonl_to_json(input_file, output_file)
    except Exception as e:
        print(f"错误: {e}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())