# merge_lora_weights.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

print("加载基础模型和分词器...")
base_model = AutoModelForCausalLM.from_pretrained("./Llama-2-7b-hf/")
tokenizer = AutoTokenizer.from_pretrained("./Llama-2-7b-hf/")

print("加载LoRA适配器...")
model = PeftModel.from_pretrained(base_model, "./finetuned/")

print("合并权重...")
merged_model = model.merge_and_unload()

print("保存合并后的模型和分词器...")
merged_model.save_pretrained("./finetuned_merged/")
tokenizer.save_pretrained("./finetuned_merged/")
