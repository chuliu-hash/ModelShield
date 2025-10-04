#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统计没有在查询sentence中出现的水印词watermark_words,写入新字段not_in_query_wm
"""

import json
import argparse


def extract_not_in_query_watermarks(input_file, output_file, check_human=False):
    """
    提取未在查询中出现的水印词

    Args:
        input_file: 输入JSON文件路径
        output_file: 输出JSON文件路径
        check_human: 是否同时检查human_answers字段
    """
    # 读取数据
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 处理每条数据
    for item in data:
        sentence = item.get('sentence', '')
        watermark_words = item.get('watermark_words', [])

        # 统计未在sentence中出现的水印词
        not_in_query_wm = []
        for word in watermark_words:
            if word not in sentence:
                not_in_query_wm.append(word)

        # 添加新字段
        item['not_in_query_wm'] = not_in_query_wm

        # 如果需要检查human_answers
        if check_human:
            human_answers = item.get('human_answers', '')
            not_in_query_and_human = []
            for word in watermark_words:
                if word not in sentence and word not in human_answers:
                    not_in_query_and_human.append(word)
            item['not_in_query_and_human'] = not_in_query_and_human

    # 写入结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # 统计信息
    total_items = len(data)
    total_watermarks = sum(len(item['watermark_words']) for item in data)
    total_not_in_query = sum(len(item['not_in_query_wm']) for item in data)

    print(f"处理完成！")
    print(f"总条目数: {total_items}")
    print(f"总水印词数: {total_watermarks}")
    print(f"未在查询中出现的水印词数: {total_not_in_query}")
    print(f"未出现比例: {total_not_in_query/total_watermarks*100:.2f}%")

    if check_human:
        total_not_in_query_and_human = sum(len(item['not_in_query_and_human']) for item in data)
        print(f"未在查询和人工回答中都出现的水印词数: {total_not_in_query_and_human}")
        print(f"未在查询和人工回答中都出现比例: {total_not_in_query_and_human/total_watermarks*100:.2f}%")

    print(f"输出文件: {output_file}")


if __name__ == '__main__':
    input_file ='finetuned_output.json'
    output_file = 'finetuned_output.json'
    check_human = True  # 设置为True检查human_answers，False只检查sentence
    extract_not_in_query_watermarks(input_file, output_file, check_human)