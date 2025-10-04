#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 human_answers 从列表转换为字符串
由于每个 human_answers 只包含一个元素，将其转换为字符串格式
"""

import json
import os

def convert_human_answers_to_string(input_file, output_file):
    """
    将 human_answers 从列表转换为字符串

    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
    """
    print(f"正在读取 {input_file}...")

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"文件不存在: {input_file}")

    # 读取数据
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"读取到 {len(data)} 条记录")

    # 统计
    converted_count = 0
    multi_answers_count = 0

    # 转换
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

        if (i + 1) % 10 == 0:
            print(f"已处理 {i + 1} 条记录...")

    print(f"转换完成，共转换 {converted_count} 条记录")
    print(f"其中有多条human_answers的记录: {multi_answers_count} 条（已选择最长的一条）")

    # 保存结果
    print(f"正在保存到 {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("保存完成！")

    # 验证
    print("\n验证转换结果:")
    sample_indices = [0, 9, 34] if len(data) > 34 else [0]
    for idx in sample_indices:
        if idx < len(data):
            print(f"\n记录 {idx}:")
            print(f"  human_answers 类型: {type(data[idx].get('human_answers', ''))}")
            print(f"  human_answers 前100字符: {str(data[idx].get('human_answers', ''))[:100]}")

def main():
    """主函数"""
    input_file = '../Watermark Verification/data/finetuned_output.json'
    output_file = '../Watermark Verification/data/finetuned_output.json'

    # 备份原文件
    backup_file = 'finetuned_output.json.backup'
    if not os.path.exists(backup_file):
        print(f"创建备份文件: {backup_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        with open(backup_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print("备份完成")

    try:
        convert_human_answers_to_string(input_file, output_file)
        print("\n=== 转换成功 ===")
        return 0
    except Exception as e:
        print(f"\n程序执行出错: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
