#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从json中提取sentence和prediction_wm字段
"""

import json

# 读取JSON文件
print("正在读取文件...")
with open('finetuned_llama2.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 提取sentence和prediction_wm字段
print(f"开始提取字段,共有{len(data)}条记录...")
extracted_data = []

for item in data:
    extracted_item = {
        'sentence': item.get('sentence', ''),
        'prediction_wm': item.get('prediction_wm', '')
    }
    extracted_data.append(extracted_item)

# 保存结果
output_file = 'extracted_sentence_prediction.json'
print(f"正在保存到{output_file}...")
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(extracted_data, f, ensure_ascii=False, indent=2)

print("完成!")
print(f"成功提取{len(extracted_data)}条记录")
print(f"结果已保存到: {output_file}")
