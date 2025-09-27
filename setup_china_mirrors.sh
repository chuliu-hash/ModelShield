#!/bin/bash
# ModelShield 国内镜像源安装脚本

echo "🚀 配置国内镜像源并安装ModelShield环境..."

# 1. 配置conda国内镜像源
echo "📦 配置Conda镜像源..."
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/nvidia/
conda config --set show_channel_urls yes

# 2. 配置pip国内镜像源
echo "🐍 配置pip镜像源..."
mkdir -p ~/.pip
cat > ~/.pip/pip.conf << EOF
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple/
trusted-host = pypi.tuna.tsinghua.edu.cn
timeout = 120

[install]
trusted-host = pypi.tuna.tsinghua.edu.cn
EOF

# 3. 创建conda环境
echo "🌟 创建ModelShield环境..."
conda env create -f environment_china.yml

# 4. 激活环境并验证安装
echo "✅ 验证安装..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate modelshield

echo "🔍 检查PyTorch安装..."
python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'CUDA版本: {torch.version.cuda}')"

echo "🔍 检查Transformers安装..."
python -c "import transformers; print(f'Transformers版本: {transformers.__version__}')"

echo "🔍 检查其他关键库..."
python -c "
try:
    import datasets, accelerate, peft, deepspeed
    print('✅ 所有关键库安装成功!')
    print(f'datasets: {datasets.__version__}')
    print(f'accelerate: {accelerate.__version__}')
    print(f'peft: {peft.__version__}')
    print(f'deepspeed: {deepspeed.__version__}')
except ImportError as e:
    print(f'❌ 导入错误: {e}')
"

echo ""
echo "🎉 环境安装完成！"
echo "💡 使用方法："
echo "   conda activate modelshield"
echo ""
echo "🔧 如果遇到网络问题，可以尝试以下镜像源："
echo "   清华源: https://mirrors.tuna.tsinghua.edu.cn/"
echo "   中科大源: https://mirrors.ustc.edu.cn/"
echo "   阿里云源: https://mirrors.aliyun.com/"