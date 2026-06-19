#!/bin/bash
set -e

# 1. Install dependencies
pip install -r requirements.txt

# 2. HuggingFace login (required for gated models like MedGemma)
# Get your token at https://huggingface.co/settings/tokens
hf auth login --token "$HF_TOKEN"

# 3. Download datasets
echo "Downloading OmniMedVQA..."
python -c "from utils import load_dataset; load_dataset()"

echo "Downloading PMC-VQA..."
python -c "from utils import create_pmcvqa_dataset; create_pmcvqa_dataset()"

echo "Downloading MedXpertQA..."
python -c "from dataset_providers import create_medxpertqa_dataset; create_medxpertqa_dataset()"

# 4. Download model weights
echo "Downloading MedGemma-4B-IT..."
hf download "google/medgemma-4b-it"

echo "Downloading Qwen2-VL-7B-Instruct..."
hf download "Qwen/Qwen2-VL-7B-Instruct"

# 5. (Optional) WandB login for training logging
# wandb login "$WANDB_API_KEY"
