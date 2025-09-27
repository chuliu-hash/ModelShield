@echo off
REM LoRA微调训练脚本 (Windows版本)
REM 使用LoRA (Low-Rank Adaptation) 进行高效微调，大幅减少GPU内存需求

set CUDA_VISIBLE_DEVICES=0,1,2,3
set WANDB_DISABLED=true

REM 训练参数
set MODEL_CONFIG=config/Llam2.json
set LORA_CONFIG=config/lora_config_llama.json

echo 开始LoRA微调训练...
echo 模型配置: %MODEL_CONFIG%
echo LoRA配置: %LORA_CONFIG%

python finetune_imitation_model_my.py ^
    --model_config_file %MODEL_CONFIG% ^
    --use_lora ^
    --lora_hyperparams_file %LORA_CONFIG% ^
    --resume_from_checkpoint false

echo LoRA训练完成！
pause