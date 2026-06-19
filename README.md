# Verbalized Uncertainty Calibration of Multimodal LLMs in Medical VQA

This repository contains the code for fine-tuning multimodal LLMs (MedGemma-4B-IT, Qwen2-VL-7B-Instruct) to produce calibrated verbalized confidence scores on medical VQA tasks.

The model outputs a confidence score alongside its answer. Training uses a composite loss:

**L = lambda * L_Brier + (1 - lambda) * L_Anchor + alpha * L_Align + beta * L_KL**

where L_Brier calibrates verbalized confidence, L_Anchor preserves answer quality, L_Align enforces causal consistency across a 2x2 perturbation design (image presence x text integrity), and L_KL regularizes answer-token logits via top-k KL divergence against the frozen base model.

## Setup

```bash
export HF_TOKEN="your_huggingface_token"  # Required for gated models (MedGemma)
bash setup.sh
```

This installs dependencies, downloads the OmniMedVQA dataset, and pre-downloads MedGemma weights. A HuggingFace token with access to `google/medgemma-4b-it` is required (request access at the model page).

For Qwen2-VL, weights are downloaded automatically on first use (no gating).

## Data

Some underlying datasets (e.g., RadImageNet) are access-restricted, so we ship only
the OmniMedVQA `question_id`s rather than redistributing content. The pipeline
downloads OmniMedVQA and reconstructs the full splits automatically.

## Reproducing the Pipeline

### Step 0: Perturbation Data Collection

Run base model inference on the sample pool under 4 perturbation conditions (2x2 factorial: image present/absent x text intact/corrupted), each with 10 runs at temperature 1.

First, save prompt templates:

```bash
python inference_vllm_medgemma.py \
    --dataset omnimed_train \
    --build-prompts-only
```

Then run inference (10 runs, temp=1):

```bash
python inference_vllm_medgemma.py \
    --dataset omnimed_train \
    --num_runs 10 \
    --temperature 1
```

This produces `.npy` files for all 4 perturbation methods:

- `verbalized` (real image + original text)
- `visual_contrast` (black image + original text)
- `verbalized_shuffle_replaced_ts` (real image + corrupted text)
- `visual_contrast_shuffle_replaced_ts` (black image + corrupted text)

For PMC-VQA, replace `--dataset omnimed_train` with `--dataset pmcvqa_train` and use `--prefix pmcvqa_train_medgemma_` in Step 1.

### Step 1: Training Dataset Creation

Build the grouped training CSV from the inference outputs:

```bash
python create_dataset.py \
    --data_dir ./ \
    --output training_dataset_4k.csv
```

This produces a CSV with columns `[group_id, full_text, image_path, correct]` where each group contains 4 rows (one per perturbation condition) and `correct` is the mean accuracy across 10 inference runs.

### Step 2: Training

**MedGemma-4B-IT:**

```bash
python train_medgemma.py \
    --dataset_file training_dataset_4k.csv \
    --output_dir ./outputs_medgemma \
    --num_epochs <epochs> \
    --lr <learning_rate> \
    --lora_rank <rank> --lora_alpha <lora_scaling> \
    --brier_anchor_lambda <lambda> \
    --egc_alpha <alpha> \
    --use_kl_anchor \
    --kl_weight <beta> \
    --kl_top_k <top_k>
```

**Qwen2-VL-7B-Instruct:**

```bash
python train_qwen.py \
    --dataset_file training_dataset_4k.csv \
    --output_dir ./outputs_qwen2 \
    --num_epochs <epochs> \
    --lr <learning_rate> \
    --lora_rank <rank> --lora_alpha <lora_scaling> \
    --brier_anchor_lambda <lambda> \
    --egc_alpha <alpha> \
    --use_kl_anchor \
    --kl_weight <beta> \
    --kl_top_k <top_k>
```

Key hyperparameters:

- `--brier_anchor_lambda` (lambda): Brier vs anchor loss mixing weight
- `--egc_alpha` (alpha): Alignment loss weight
- `--kl_weight` (beta): KL divergence regularization weight
- `--kl_top_k`: Number of top tokens for KL computation

See the paper for specific hyperparameter values per configuration.

**Replicating baselines:**

ConfTuner baseline can be reproduced using the same training scripts with:

```bash
python train_medgemma.py \
    --dataset_file training_dataset_4k.csv \
    --output_dir ./outputs_conftuner \
    --ungrouped \
    --add_loss_con \
    --brier_anchor_lambda 1.0 \
    --egc_alpha 0
```

### Step 3: Evaluation

**MedGemma (HF or vLLM):**

```bash
# Option A: HF inference (loads LoRA directly, no merge step)
python eval_model.py \
    --model_path ./outputs_medgemma/models/<run_name>_epoch_2

# Option B: vLLM inference (merges LoRA first, faster for large-scale eval)
python merge_and_eval_medgemma.py \
    --lora_path ./outputs_medgemma/models/<run_name>_epoch_2
```

**Qwen2-VL:**

```bash
python merge_and_eval.py \
    --lora_path ./outputs_qwen2/models/<run_name>_epoch_3
```

Both scripts run inference on OmniMedVQA, PMC-VQA, and MedXpertQA, then report Accuracy, Brier Score, ECE, and AUROC.

## Evaluation Benchmarks

| Benchmark  | Type    | Source             |
| ---------- | ------- | ------------------ |
| OmniMedVQA | ID/OOD  | Hu et al., 2024    |
| PMC-VQA    | ID/OOD  | Zhang et al., 2024 |
| MedXpertQA | OOD     | Zuo et al., 2025   |
