#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将nowm.json的prediction_nowm插入到wm.json相同sentence记录处
位于prediction_wm的下一个位置
"""

import json
from collections import OrderedDict

# 读取两个JSON文件
print("正在读取文件...")
with open('finetuned_llama2.json', 'r', encoding='utf-8') as f:
    llama2_data = json.load(f)

with open('finetuned_output_nowm.json', 'r', encoding='utf-8') as f:
    nowm_data = json.load(f)

# 创建一个字典,以sentence为key,方便查找
nowm_dict = {item['sentence']: item['prediction_nowm'] for item in nowm_data}

# 合并数据
print(f"开始合并数据,共有{len(llama2_data)}条记录...")
merged_count = 0

for item in llama2_data:
    sentence = item['sentence']
    if sentence in nowm_dict:
        # 创建新的有序字典,保证prediction_nowm在prediction_wm之后
        new_item = OrderedDict()
        for key, value in item.items():
            new_item[key] = value
            # 在prediction_wm之后插入prediction_nowm
            if key == 'prediction_wm':
                new_item['prediction_nowm'] = nowm_dict[sentence]

        # 用新的有序字典替换原item的所有键值
        item.clear()
        item.update(new_item)
        merged_count += 1

print(f"成功合并{merged_count}条记录")

# 保存结果
output_file = 'finetuned_llama2_merged.json'
print(f"正在保存到{output_file}...")
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(llama2_data, f, ensure_ascii=False, indent=2)

print("完成!")
print(f"结果已保存到: {output_file}")
