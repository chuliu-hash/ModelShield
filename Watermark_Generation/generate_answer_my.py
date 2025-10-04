from openai import OpenAI, APIError
import json
import os
from time import sleep

api_keys=["sk-hVEye8BA2688bc89BCB2T3BLbKFJ1d342476887D45658Bbf",]
base_url="https://c-z0-api-01.hash070.com/v1/chat/completions"
filepath = "output/HC3_watermarked_zh.json"

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

wildwind_data = read_list_from_file("input/HC3_watermarked_zh.json")[max_id:]

cnt = 0

for item in wildwind_data:
  

    system_prompt = """我会问你一些问题，请你按照json的格式输出回答，格式如下:
              {"answer": "..."}"""


    messages = [
    {"role": "system", "content": system_prompt.strip()},  # 系统级指令
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
            print(f"Query:{item['sentence']}\nAnswer:{answer_data['answer']}\n")
            
            output.append({
                "id": item["id"],
                "query": item["sentence"],
                "prediction_noWM": answer_data["answer"],
                "prediction_WM": item["prediction"],
            })
            
                
    except APIError as e:
        print(f"API Error: {e}")
        with open(filepath, "w", encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=4)
    finally:
        cnt += 1

    sleep(1)

    
with open(filepath, "w", encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=4)