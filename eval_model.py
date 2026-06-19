#!/usr/bin/env python3
"""
Evaluation script for trained models.

Given a model path like:
    /training_outputs_kl/models/exp2_lr1e5_r8a32_ra5_rw1_kl250_epoch_3

Or a wandb artifact like:
    project_name/model-run_name-epoch-1:latest

Runs inference on omnimed, pmcvqa, and medxpertqa datasets, then computes and reports:
    - Brier Score
    - AUROC
    - ECE
    - Accuracy

Usage:
    python eval_model.py --model_path ./training_outputs_kl/models/exp2_lr1e5_r8a32_ra5_rw1_kl250_epoch_3
    python eval_model.py --wandb_artifact myproject/model-run_name-epoch-1:latest
"""

import argparse
import subprocess
import os
import sys
import pandas as pd
import numpy as np

from utils import compute_brier, compute_ece, compute_auroc, load_method_run0


def download_wandb_artifact(artifact_name: str, download_dir: str = "./wandb_artifacts") -> str:
    """Download a model artifact from wandb and return the local path.
    
    Args:
        artifact_name: Full artifact name (e.g., 'project/artifact-name:version')
        download_dir: Directory to download artifacts to
        
    Returns:
        Local path to the downloaded artifact
    """
    try:
        import wandb
    except ImportError:
        raise ImportError("wandb is required for artifact downloading. Install with: pip install wandb")
    
    print(f"Downloading wandb artifact: {artifact_name}")
    
    # Initialize wandb API (uses existing login)
    api = wandb.Api()
    
    # Get the artifact
    artifact = api.artifact(artifact_name)
    
    # Download to local directory
    artifact_dir = artifact.download(root=download_dir)
    
    print(f"Artifact downloaded to: {artifact_dir}")
    
    return artifact_dir


def run_inference(model_path: str, output_prefix: str, dataset: str, filter_from: str = None, batch_size: int = None, num_runs: int = None, no_compile: bool = False, inference_script: str = "inference_hf_medgemma.py", quick_test: bool = False):
    """Run inference script for a given dataset."""
    cmd = [
        sys.executable, inference_script,
        "--lora_path", model_path,
        "--output_prefix", f"{output_prefix}_",
        "--dataset", dataset,
        "--greedy",
    ]
    if batch_size:
        cmd.extend(["--batch_size", str(batch_size)])
    if num_runs:
        cmd.extend(["--num_runs", str(num_runs)])
    if no_compile:
        cmd.append("--no_compile")
    if filter_from and dataset == "omnimed":
        cmd.extend(["--filter_from", filter_from])
    if quick_test:
        cmd.append("--quick_test")
    print(f"\n{'='*60}")
    print(f"Running inference for dataset: {dataset}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    subprocess.run(cmd, check=True)


def compute_metrics(output_prefix: str, dataset: str, score_divisor: float = 10, model_tag: str = "medgemma") -> dict:
    """Load inference outputs and compute metrics."""
    method_name = f"{output_prefix}_{dataset}_{model_tag}_verbalized"
    temp = 0

    # Load parsed results
    df, picked, gens, gts = load_method_run0(method_name, temp, num_runs_candidates=(1, 10, 5))

    # Extract arrays for metric computation
    y_true = df['correct'].astype(int).to_numpy()
    y_prob = (df['confidence'] / score_divisor).to_numpy()

    # Compute metrics
    accuracy = y_true.mean()
    brier = compute_brier(y_true, y_prob)
    ece, _, _ = compute_ece(y_true, y_prob, source_scale=score_divisor)
    auroc = compute_auroc(y_true, y_prob)

    return {
        'dataset': dataset,
        'accuracy': accuracy,
        'brier': brier,
        'ece': ece,
        'auroc': auroc,
        'n_samples': len(y_true),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model on multiple datasets")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to the trained model (LoRA adapter)")
    parser.add_argument("--wandb_artifact", type=str, default=None,
                        help="Wandb artifact name (e.g., 'project/model-name:version')")
    parser.add_argument("--artifact_download_dir", type=str, default="./wandb_artifacts",
                        help="Directory to download wandb artifacts to")
    parser.add_argument("--skip_omnimed", action="store_true",
                        help="Skip OmniMedVQA dataset evaluation")
    parser.add_argument("--skip_pmcvqa", action="store_true",
                        help="Skip PMC-VQA dataset evaluation")
    parser.add_argument("--skip_medxpertqa", action="store_true",
                        help="Skip MedXpertQA dataset evaluation")
    parser.add_argument("--skip_omnimed_test", action="store_true",
                        help="Skip OmniMedVQA IID test set evaluation")
    parser.add_argument("--filter_from", type=str, default=None,
                        help="Path to training dataset CSV to filter OmniMedVQA test set (uses same samples as training)")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size for inference")
    parser.add_argument("--num_runs", type=int, default=None,
                        help="Number of inference runs per sample (default: method default)")
    parser.add_argument("--no_compile", action="store_true",
                        help="Skip torch.compile() to save memory")
    parser.add_argument("--quick_test", action="store_true",
                        help="Limit to a small subset for a fast sanity run")
    parser.add_argument("--inference_script", type=str, default="inference_hf_medgemma.py",
                        help="Inference script to use (default: inference_hf_medgemma.py)")
    parser.add_argument("--score_divisor", type=float, default=10,
                        help="Confidence scale divisor for normalization (default: 10 for MedGemma 1-10, use 9 for Qwen2 0-9)")
    parser.add_argument("--model_tag", type=str, default="medgemma",
                        help="Model identifier for method name pattern (default: medgemma, use 'qwen' for Qwen2-VL)")
    args = parser.parse_args()
    
    # Validate arguments: must specify exactly one of model_path or wandb_artifact
    if args.model_path is None and args.wandb_artifact is None:
        parser.error("Must specify either --model_path or --wandb_artifact")
    if args.model_path is not None and args.wandb_artifact is not None:
        parser.error("Cannot specify both --model_path and --wandb_artifact")
    
    # Get model path (download from wandb if needed)
    if args.wandb_artifact:
        model_path = download_wandb_artifact(args.wandb_artifact, args.artifact_download_dir)
        # Extract a clean name from the artifact for output prefix
        output_prefix = args.wandb_artifact.replace("/", "_").replace(":", "_")
    else:
        model_path = args.model_path
        output_prefix = os.path.basename(model_path.rstrip('/'))
    
    print(f"\n{'#'*60}")
    print(f"Model Evaluation Script")
    print(f"{'#'*60}")
    if args.wandb_artifact:
        print(f"Wandb artifact: {args.wandb_artifact}")
    print(f"Model path: {model_path}")
    print(f"Output prefix: {output_prefix}")
    
    # Build dataset list based on skip flags
    all_datasets = [
        ("omnimed", args.skip_omnimed),
        ("pmcvqa", args.skip_pmcvqa),
        ("medxpertqa", args.skip_medxpertqa),
        ("omnimed_test", args.skip_omnimed_test),
    ]
    datasets = [name for name, skip in all_datasets if not skip]
    skipped = [name for name, skip in all_datasets if skip]
    
    if skipped:
        print(f"Datasets: {', '.join(datasets)} (skipping {', '.join(skipped)})")
    else:
        print(f"Datasets: {', '.join(datasets)}")
    if args.filter_from:
        print(f"OmniMedVQA filter: {args.filter_from}")
    print(f"{'#'*60}\n")
    
    results = []
    
    # Run inference and compute metrics for each dataset
    for dataset in datasets:
        run_inference(model_path, output_prefix, dataset, filter_from=args.filter_from, batch_size=args.batch_size, num_runs=args.num_runs, no_compile=args.no_compile, inference_script=args.inference_script, quick_test=args.quick_test)
        metrics = compute_metrics(output_prefix, dataset, score_divisor=args.score_divisor, model_tag=args.model_tag)
        results.append(metrics)
        
        print(f"\n--- Results for {dataset} ---")
        print(f"  Accuracy: {metrics['accuracy']:.3f}")
        print(f"  Brier:    {metrics['brier']:.3f}")
        print(f"  ECE:      {metrics['ece']:.3f}")
        print(f"  AUROC:    {metrics['auroc']:.3f}")
        print(f"  Samples:  {metrics['n_samples']}")
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df['model'] = output_prefix
    
    # Reorder columns
    results_df = results_df[['model', 'dataset', 'n_samples', 'accuracy', 'brier', 'ece', 'auroc']]
    
    csv_path = f"{output_prefix}_eval_results.csv"
    results_df.to_csv(csv_path, index=False)
    
    print(f"\n{'='*60}")
    print(f"Evaluation Complete!")
    print(f"{'='*60}")
    print(f"\nResults saved to: {csv_path}")
    print(f"\nSummary:")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
