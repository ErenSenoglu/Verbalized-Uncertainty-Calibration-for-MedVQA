#!/usr/bin/env python3
"""
Create a grouped perturbation training dataset from inference outputs.

This script takes the .npy outputs from Step 0 (4 perturbation conditions x 10 runs)
and produces a grouped CSV suitable for training with train_medgemma.py / train_qwen.py.

The 4 perturbation conditions (2x2 factorial design):
    1. verbalized                              — real image + original text
    2. visual_contrast                         — black image + original text
    3. verbalized_shuffle_replaced_ts          — real image + corrupted text
    4. visual_contrast_shuffle_replaced_ts     — black image + corrupted text

Prerequisites:
    For each method, you need two sets of .npy files produced by inference_vllm_medgemma.py:
        1. Run with --build-prompts-only to save prompt templates:
               {prefix}{method}_prompts.npy
               {prefix}{method}_ground_truths.npy
        2. Run inference normally (10 runs, temp=1):
               {prefix}{method}_all_generations_10runs_temp1.npy
               {prefix}{method}_ground_truths.npy

Usage:
    python create_dataset.py \\
        --data_dir /path/to/npy/files \\
        --output training_dataset_4k.csv
"""

import argparse
import os
import re

import numpy as np
import pandas as pd

from utils import load_method_run10


METHODS = [
    "verbalized",
    "visual_contrast",
    "verbalized_shuffle_replaced_ts",
    "visual_contrast_shuffle_replaced_ts",
]


def trim_after_confidence(text_array):
    """Trim text to include everything up to and including 'Confidence: '.

    The training target is the confidence digit; full_text ends right before it.
    """
    trimmed = []
    for text in text_array:
        match = re.search(r"Confidence:", text, re.IGNORECASE)
        if match:
            trimmed.append(text[: match.end() + 1])  # +1 for trailing space
        else:
            trimmed.append(text)
    return trimmed


def build_method_dataframe(method, indices, data_dir, prefix, black_image_path, num_runs=10, temp_str="1"):
    """Build a DataFrame for one perturbation method, filtered to selected indices.

    Returns DataFrame with columns: full_text, image_path, correct
    """
    # Load prompts (saved via --build-prompts-only)
    prompts_path = os.path.join(data_dir, f"{prefix}{method}_prompts.npy")
    prompts = np.load(prompts_path, allow_pickle=True)

    # Load generations
    gens_path = os.path.join(data_dir, f"{prefix}{method}_all_generations_{num_runs}runs_temp{temp_str}.npy")
    generations = np.load(gens_path, allow_pickle=True)
    generations_first = generations[:, 0]

    # Compute mean accuracy across runs via load_method_run10
    results = load_method_run10(
        f"{prefix}{method}", temp=float(temp_str.replace("_", ".")), num_runs_candidates=(num_runs,), relative_path=data_dir
    )
    mean_correct = results.groupby("sample_idx")["correct"].mean().values

    # Filter to selected indices
    prompts_filtered = prompts[indices]
    generations_filtered = generations_first[indices]
    correct_filtered = mean_correct[indices]

    # Build dataframe from prompt dicts
    df = pd.DataFrame(prompts_filtered.tolist())
    df["image_path"] = df["multi_modal_data"].apply(lambda x: x["image"])
    df = df.drop(columns=["multi_modal_data"])
    df["correct"] = correct_filtered
    df["full_text"] = df["prompt"] + trim_after_confidence(generations_filtered)
    df = df.drop(columns=["prompt"])

    # Visual contrast methods use a black image
    if method.startswith("visual_contrast"):
        df["image_path"] = black_image_path

    return df[["full_text", "image_path", "correct"]]


def main():
    parser = argparse.ArgumentParser(
        description="Create grouped perturbation training dataset from inference outputs"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Directory containing .npy files from inference",
    )
    parser.add_argument(
        "--prefix", type=str, default="omnimed_train_medgemma_",
        help="Filename prefix for .npy files (default: omnimed_train_medgemma_)",
    )
    parser.add_argument(
        "--num_runs", type=int, default=10,
        help="Number of inference runs in the .npy files (default: 10)",
    )
    parser.add_argument(
        "--temperature", type=float, default=1,
        help="Temperature used during inference (default: 1)",
    )
    parser.add_argument(
        "--output", type=str, default="training_dataset.csv",
        help="Output CSV path (default: training_dataset.csv)",
    )
    parser.add_argument(
        "--black_image_path", type=str,
        default="./OmniMedVQA_data/OmniMedVQA/black.png",
        help="Path to black image for visual_contrast conditions",
    )
    args = parser.parse_args()

    # Ensure data_dir path ends with separator for load_method_run10
    data_dir = args.data_dir
    if not data_dir.endswith(os.sep):
        data_dir += os.sep

    # Create black image for visual_contrast conditions if it doesn't exist
    black_image_path = args.black_image_path
    if not os.path.exists(black_image_path):
        from PIL import Image
        os.makedirs(os.path.dirname(black_image_path) or ".", exist_ok=True)
        Image.new("RGB", (224, 224), (0, 0, 0)).save(black_image_path)
        print(f"Created black image at {black_image_path}")

    # Use all samples from the .npy files
    # Format temperature string same way as inference script
    temp_str = str(args.temperature).replace('.', '_')

    first_method = METHODS[0]
    gens_path = os.path.join(data_dir, f"{args.prefix}{first_method}_all_generations_{args.num_runs}runs_temp{temp_str}.npy")
    n_total = np.load(gens_path, allow_pickle=True).shape[0]
    selected_indices = np.arange(n_total)
    print(f"Using all {n_total} samples from {gens_path}")

    # Build per-method dataframes
    method_dfs = []
    for method in METHODS:
        print(f"  Processing {method}...")
        df_method = build_method_dataframe(
            method, selected_indices, data_dir, args.prefix, args.black_image_path, args.num_runs, temp_str
        )
        method_dfs.append(df_method)

    # Interleave: all 4 methods for sample 0, then sample 1, etc.
    combined_rows = []
    for i in range(len(method_dfs[0])):
        for df in method_dfs:
            combined_rows.append(df.iloc[i])

    combined_df = pd.DataFrame(combined_rows).reset_index(drop=True)
    combined_df["group_id"] = combined_df.index // 4

    # Reorder columns
    combined_df = combined_df[["group_id", "full_text", "image_path", "correct"]]

    # Save
    combined_df.to_csv(args.output, index=False)
    print(f"\nSaved {len(combined_df)} rows ({n_total} groups x 4 perturbations)")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
