@echo off
REM DeepSpeed分布式训练脚本 (Windows版本)
REM 使用DeepSpeed ZeRO优化大幅减少GPU内存需求

set WANDB_DISABLED=true

REM 配置文件
set MODEL_CONFIG=config/llama2.json
set DEEPSPEED_CONFIG=config/deepspeed_config.json

REM GPU数量
set NUM_GPUS=4

echo 开始DeepSpeed训练...
echo GPU数量: %NUM_GPUS%
echo 模型配置: %MODEL_CONFIG%
echo DeepSpeed配置: %DEEPSPEED_CONFIG%

REM 使用deepspeed启动训练
deepspeed --num_gpus=%NUM_GPUS% finetune_imitation_model_my.py ^
    --model_config_file %MODEL_CONFIG% ^
    --deepspeed %DEEPSPEED_CONFIG%

echo DeepSpeed训练完成！
pause