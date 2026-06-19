"""MedGemma + OmniMedVQA specific utilities.

This module contains helper functions for building prompts and confidence
targets for MedGemma-4B-IT fine-tuning on OmniMedVQA.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


from utils import (
    load_dataset as _helpers_load_dataset,
    load_method_run0,
    compute_brier,
    compute_ece,
    compute_auroc,
    plot_ece_diagrams,
)


def compute_metrics(method_name: str, temp: float = 0, relative_path: str = "",
                    source_scale: int = 10, n_bins: int = 10,
                    num_runs_candidates=(1,)):
    """Load a method's run-0 results and compute all calibration metrics.

    Args:
        method_name: Prefix for npy files (e.g. "pmc_v7_l0.9_a2.0_e2_omnimed_test_medgemma_verbalized").
        temp: Temperature used during generation.
        relative_path: Directory containing the npy files.
        source_scale: Confidence scale (10 for 1-10 scores).
        n_bins: Number of bins for ECE.
        num_runs_candidates: Tuple of candidate run counts to try.

    Returns:
        dict with keys: accuracy, brier, auroc, ece, mean_conf, n, df
    """
    df, _, _, _ = load_method_run0(
        method_name, temp=temp,
        num_runs_candidates=num_runs_candidates,
        relative_path=relative_path,
    )
    if df is None:
        print(f"[ERROR] Could not load {method_name}")
        return None

    acc = df["correct"].astype(float).values
    conf = df["confidence"].values / (source_scale if source_scale == 10 else 1)

    accuracy = float(acc.mean())
    mean_conf = float(conf.mean())
    brier = compute_brier(acc, conf)
    auroc = compute_auroc(acc, conf)
    ece, bin_counts, bin_accuracies = compute_ece(acc, conf, source_scale=source_scale, n_bins=n_bins)

    print(f"\n{'='*50}")
    print(f"  {method_name}")
    print(f"{'='*50}")
    print(f"  N         = {len(acc)}")
    print(f"  Accuracy  = {accuracy:.4f}")
    print(f"  Brier     = {brier:.4f}")
    print(f"  AUROC     = {auroc:.4f}")
    print(f"  ECE       = {ece:.4f}")
    print(f"  Mean Conf = {mean_conf:.4f}")

    plot_ece_diagrams(
        acc,
        {"verbalized_confidence": conf},
        source_scale=source_scale,
        n_bins=n_bins,
        model_name=method_name,
    )

    return {
        "accuracy": accuracy,
        "brier": brier,
        "auroc": auroc,
        "ece": ece,
        "mean_conf": mean_conf,
        "n": len(acc),
        "df": df,
    }

def load_training_set(dataset_name) -> pd.DataFrame:
    """Load the training DataFrame for a given dataset.

    This simply delegates to :func:`helpers.load_dataset` and
    :func:`helpers.create_dataset` so that any filtering or path
    logic stays in one place.

    Args:
        dataset_name: One of 'omnimedvqa', 'nejm', or 'pmcvqa'.

    Returns:
        pd.DataFrame: Filtered DataFrame with columns: group_id, full_text, image_path, correct.
    """

    # Ensure the raw dataset is present (no-op if already extracted)
    if dataset_name == "omnimedvqa":
        _helpers_load_dataset()
        df = pd.read_csv("medgemma-4b-it_omnimedvqa_perturbation_dataset.csv")
    elif dataset_name == "pmcvqa":
        df = pd.read_csv("medgemma-4b-it_pmcvqa_perturbation_dataset.csv")
    else:
        raise RuntimeError(f"Unknown dataset name: {dataset_name}. Supported: omnimedvqa, pmcvqa")
        
    if df is None:
        raise RuntimeError(f"Failed to load training data for {dataset_name}.")
    
    return df