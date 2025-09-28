## Overview

ModelShield provides a framework for generating, embedding, and verifying watermarks in language models to protect intellectual property against model extraction attacks. This repository includes code for:
- Watermark generation and verification.
- Training imitation models with watermarked data and Generating imitation model’s output.
- Supporting datasets for experimentation.

---

## Stages

### 1. **Watermark Generation**
We utilize system-level instructions to guide watermark generation in language models, ensuring seamless integration and high robustness. (need API-KEY for different LMaaS)

### 2. **Imitation Model Training**
Fine-tune imitation models with watermarked data to simulate model extraction attacks. 

We base our training and fine-tuning on the [BELLE GitHub project](https://github.com/LianjiaTech/BELLE). Key features include:
- **Full fine-tuning** and **LoRA fine-tuning** support.
- Flexibility to integrate your own fine-tuning methods.
- Configuration options available in the `config` directory (e.g., model base, fine-tuning epochs, batch size, learning rate, and LoRA usage).

### 3. **Watermark Verification**
We provide two methods for verifying embedded watermarks:
1. **Rapid Verification**: Quickly detect the presence of a watermark based on texts.
2. **Detailed Verification**: Need comparison with legimate model and base model.
   
---

## Datasets

The following datasets are used in our experiments:
- **HC3**: A dataset for language model imitation analysis.
- **WILD**: A dataset for evaluating robustness in diverse scenarios.

---
## Usage
1. 1 Generate watermarked data from the victim model (See readme in Watermark Generation)
2. 2 Simulate the model extraction attack (See readme in Imitation Model training)
3. 3 Verify the watermark (See readme in Watermark Verification)

---
