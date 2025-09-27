from openai import OpenAI, APIError
import json
import os
from time import sleep

api_keys=["sk-hVEye8BA2688bc89BCB2T3BLbKFJ1d342476887D45658Bbf",]
base_url="https://c-z0-api-01.hash070.com/v1"
filepath = "output/answers.json"

output = []
max_id = 0
undo_done_id=[]

if os.path.exists(filepath):
    with open(filepath,"r",encoding='utf-8') as f:
        output = json.load(f)
    for i in output:
        undo_done_id.append(i["id"])
        if i['id'] > max_id:
            max_id = i['id']

def read_list_from_file(file_path):
    with open(file_path, "r",encoding="utf-8")as f:
        lst = eval(f.read())
    return lst

wildwind_data = read_list_from_file("input/questions.json")[max_id:]

cnt = 0

for item in wildwind_data:
  

    system_prompt = """You are a text generation model with watermarking capability.
                    Follow these rules:
                    1. Use some special words as watermarks in the generated texts.
                    2. Output the watermark words and the answer in json format:{"answer": "...", "watermark_words": [...]}
                    3. Watermark words must appear naturally in the answer"""


    messages = [
    {"role": "system", "content": system_prompt.strip()},  # 系统级水印指令
    {"role": "user", "content": item["instruction"].strip()}     # 纯问题内容
    ]

    try:
        api_key = api_keys[cnt % len(api_keys)]
        client = OpenAI(base_url=base_url, api_key=api_key)
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            response_format={"type": "json_object"}
        )
        
        if response.choices:
            answer_data = json.loads(response.choices[0].message.content)
            print(f"Query:{item['instruction']}\nAnswer:{answer_data['answer']}\nWatermark Words:{answer_data['watermark_words']}\n")
            
            output.append({
                "id": item["id"],
                "query": item["instruction"],
                "answer": answer_data["answer"],
                "watermark_words": answer_data["watermark_words"],
            })
            
                
    except APIError as e:
        print(f"API Error: {e}")
        with open(filepath, "w", encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=4)
    finally:
        cnt += 1

    sleep(5)

    
with open(filepath, "w", encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=4)