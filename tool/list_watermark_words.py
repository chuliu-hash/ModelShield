#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Watermark Words Processor Script
处理JSON文件中watermark_words字段的Python脚本
将逗号分隔的watermark_words字符串转换为列表形式
"""

import json
import os
from typing import List, Dict, Any


def process_watermark_words(input_file: str, output_file: str = None) -> None:
    """
    处理JSON文件中的watermark_words字段，将逗号分隔的字符串转换为列表

    Args:
        input_file (str): 输入的JSON文件路径
        output_file (str): 输出的JSON文件路径，如果为None则覆盖原文件
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"输入文件不存在: {input_file}")

    if output_file is None:
        # 如果未指定输出文件，则覆盖原文件
        output_file = input_file

    processed_count = 0
    total_count = 0

    try:
        # 读取JSON文件
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 确保数据是列表格式
        if not isinstance(data, list):
            raise ValueError("输入文件必须包含JSON数组格式的数据")

        # 处理每个对象中的watermark_words字段
        for item in data:
            total_count += 1
            if isinstance(item, dict) and 'watermark_words' in item:
                watermark_value = item['watermark_words']

                # 如果是字符串，则进行转换
                if isinstance(watermark_value, str):
                    watermark_str = watermark_value.strip()
                    if watermark_str:
                        # 分割字符串并去除每个元素的空白字符
                        item['watermark_words'] = [word.strip() for word in watermark_str.split(',') if word.strip()]
                    else:
                        item['watermark_words'] = []
                    processed_count += 1
                # 如果已经是列表，则跳过
                elif isinstance(watermark_value, list):
                    continue
                # 其他类型的值，保持不变但记录警告
                else:
                    print(f"警告：第{total_count}条记录的watermark_words字段类型为{type(watermark_value)}，跳过处理")

        # 写入JSON文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"处理完成！")
        print(f"输入文件: {input_file}")
        print(f"输出文件: {output_file}")
        print(f"总记录数: {total_count}")
        print(f"处理的watermark_words字段数: {processed_count}")

    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        raise




def main():
    """主函数"""
    # 直接在代码中指定输入和输出文件路径
    input_file = "data.json"  # 输入文件路径
    output_file = None  # 输出文件路径，None表示覆盖原文件

    try:
        process_watermark_words(input_file, output_file)
    except Exception as e:
        print(f"错误: {e}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())