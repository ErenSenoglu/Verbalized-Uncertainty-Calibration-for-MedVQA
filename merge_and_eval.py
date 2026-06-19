#!/usr/bin/env python3
"""
Merge LoRA adapter into Qwen2-VL-7B, run vLLM inference on benchmarks, compute metrics, cleanup.

Usage:
    python merge_and_eval.py --lora_path ./outputs_qwen2_l0.9_a2.0_e3/models/run_epoch_3
"""

import argparse
import os
import sys
import shutil
import subprocess

import torch
import pandas as pd

from utils import compute_brier, compute_ece, compute_auroc, load_method_run0

BASE_MODEL = "Qwen/Qwen2-VL-7B-Instruct"
DATASETS = ["omnimed_test", "pmcvqa", "medxpertqa"]
SCORE_DIVISOR = 9


def parse_args():
    parser = argparse.ArgumentParser(description="Merge LoRA + vLLM eval for Qwen2-VL")
    parser.add_argument("--lora_path", type=str, required=True, help="Path to LoRA adapter")
    parser.add_argument("--output_prefix", type=str, default=None, help="Override output prefix")
    parser.add_argument("--keep_merged", action="store_true", help="Keep merged model after eval")
    parser.add_argument("--quick_test", action="store_true", help="Limit to 16 samples per dataset")
    return parser.parse_args()


def merge_lora(lora_path, merged_dir):
    """Merge LoRA adapter into base model and save to disk."""
    from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
    from peft import PeftModel

    print("Loading base model on CPU...")
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cpu",
    )

    print(f"Loading LoRA adapter from {lora_path}...")
    model = PeftModel.from_pretrained(base_model, lora_path)

    print("Merging weights...")
    model = model.merge_and_unload()

    print(f"Saving merged model to {merged_dir}...")
    model.save_pretrained(merged_dir)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.save_pretrained(merged_dir)
    processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)
    processor.save_pretrained(merged_dir)

    del model, base_model
    print("Merge complete.\n")


def run_inference(merged_dir, output_prefix, dataset, quick_test=False):
    """Call inference_vllm_medgemma.py with qwen2 model config."""
    cmd = [
        sys.executable, "inference_vllm_medgemma.py",
        "--model", "qwen2",
        "--model_path", merged_dir,
        "--output_prefix", f"{output_prefix}_",
        "--dataset", dataset,
        "--greedy",
        "--num_runs", "1",
        "--method", "0",
    ]
    if quick_test:
        cmd.append("--quick_test")

    print(f"\n{'='*60}")
    print(f"Running inference for dataset: {dataset}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    subprocess.run(cmd, check=True)


def compute_metrics(output_prefix, dataset):
    """Load inference outputs and compute metrics."""
    method_name = f"{output_prefix}_{dataset}_qwen2_verbalized"

    df, picked, gens, gts = load_method_run0(method_name, temp=0, num_runs_candidates=(1,))

    y_true = df['correct'].astype(int).to_numpy()
    y_prob = (df['confidence'] / SCORE_DIVISOR).to_numpy()

    return {
        'dataset': dataset,
        'accuracy': y_true.mean(),
        'brier': compute_brier(y_true, y_prob),
        'ece': compute_ece(y_true, y_prob, source_scale=SCORE_DIVISOR)[0],
        'auroc': compute_auroc(y_true, y_prob),
        'n_samples': len(y_true),
    }


def main():
    args = parse_args()

    output_prefix = args.output_prefix or os.path.basename(args.lora_path.rstrip('/'))
    merged_dir = f"./merged_temp_{output_prefix}"

    print(f"\n{'#'*60}")
    print(f"Merge & Eval: {output_prefix}")
    print(f"LoRA: {args.lora_path}")
    print(f"Datasets: {', '.join(DATASETS)}")
    print(f"{'#'*60}\n")

    # Step 1: Merge
    print(f"{'='*60}")
    print("STEP 1: Merging LoRA adapter")
    print(f"{'='*60}")
    merge_lora(args.lora_path, merged_dir)

    # Step 2: Run inference for each dataset
    for dataset in DATASETS:
        run_inference(merged_dir, output_prefix, dataset, quick_test=args.quick_test)

    # Step 3: Compute metrics
    print(f"\n{'='*60}")
    print("STEP 3: Computing metrics")
    print(f"{'='*60}")

    results = []
    for dataset in DATASETS:
        metrics = compute_metrics(output_prefix, dataset)
        results.append(metrics)

        print(f"\n--- {dataset} ---")
        print(f"  Accuracy: {metrics['accuracy']:.3f}")
        print(f"  Brier:    {metrics['brier']:.3f}")
        print(f"  ECE:      {metrics['ece']:.3f}")
        print(f"  AUROC:    {metrics['auroc']:.3f}")
        print(f"  Samples:  {metrics['n_samples']}")

    # Save summary CSV
    results_df = pd.DataFrame(results)
    results_df['model'] = output_prefix
    results_df = results_df[['model', 'dataset', 'n_samples', 'accuracy', 'brier', 'ece', 'auroc']]
    csv_path = f"{output_prefix}_eval_results.csv"
    results_df.to_csv(csv_path, index=False)

    print(f"\n{'='*60}")
    print("ALL DONE")
    print(f"{'='*60}")
    print(f"\nResults saved to: {csv_path}")
    print(f"\n{results_df.to_string(index=False)}")

    # Step 4: Cleanup
    if not args.keep_merged:
        print(f"\nCleaning up merged model at {merged_dir}...")
        shutil.rmtree(merged_dir)
        print("Cleanup complete.")
    else:
        print(f"\nMerged model kept at {merged_dir}")


if __name__ == "__main__":
    main()
