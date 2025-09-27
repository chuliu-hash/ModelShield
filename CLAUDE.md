# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
ModelShield is a research framework for protecting language models against model extraction attacks using adaptive watermarking techniques.

## Development Commands

### Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r Imitation_Model_training/train/requirements.txt
```

### Running Components

#### Watermark Generation
```bash
# Generate watermarked answers
python Watermark_Generation/generate_answer.py
```

#### Model Training
```bash
# Fine-tune imitation model (example with LLaMA2 config)
python Imitation_Model_training/train/finetune_imitation_model.py --config Imitation_Model_training/train/configs/example_llama2.json
```

#### Watermark Verification
```bash
# Rapid verification
python "Watermark Verification"/rapid.py

# Detailed verification
python "Watermark Verification"/detailed.py
```

## Key Development Notes


### Supported Models
- Primary support for transformer-based models
- Fine-tuning techniques: Full fine-tuning and LoRA
- Compatibility with OpenAI and Hugging Face model APIs

### Verification Methods
- Statistical t-test (rapid verification)
- Kolmogorov-Smirnov test (detailed verification)
- Significance threshold: p < 0.05

## Deployment
- Docker support available in `Imitation_Model_training/train/docker/Dockerfile`
- CUDA-enabled environment recommended

## Important Dependencies
- PyTorch 1.13.0+
- Transformers 4.28.1+
- PEFT/LoRA
- DeepSpeed 0.9.0+