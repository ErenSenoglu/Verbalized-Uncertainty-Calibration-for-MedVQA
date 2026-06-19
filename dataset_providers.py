"""
External dataset providers for inference scripts.

This module provides dataset loaders that return DataFrames in the standard schema:
    question, gt_answer, image_path, option_A, option_B, option_C, option_D, [option_E], question_type

Supported datasets:
    - MedXpertQA (TsinghuaC3I/MedXpertQA): Hard medical VQA with 5 options
    - MMMU Medicine (MMMU/MMMU): Combined medicine subsets from validation split
    - Istituto dei Tumori: Lung CT MCQ dataset with 4 options
"""

import os
import re
import random
import zipfile
from typing import Optional

import pandas as pd


def create_medxpertqa_dataset(cache_dir: str = ".", seed: int = 42) -> pd.DataFrame:
    """Load MedXpertQA MM (multimodal) subset and convert to standard schema.

    MedXpertQA is a challenging medical QA dataset with 5 answer options (A-E).
    Uses the MM subset which contains images.

    Args:
        cache_dir: Directory for caching images. Images will be at cache_dir/medxpertqa_images/
        seed: Random seed for reproducibility (not used currently, kept for API consistency)

    Returns:
        pandas.DataFrame with columns: question, gt_answer, image_path,
        option_A, option_B, option_C, option_D, option_E, question_type
    """
    from datasets import load_dataset as hf_load_dataset
    from huggingface_hub import hf_hub_download

    os.makedirs(cache_dir, exist_ok=True)
    images_dir = os.path.join(cache_dir, "medxpertqa_images")

    # Download and extract images if not already done
    if not os.path.exists(images_dir):
        print("Downloading MedXpertQA images...")
        zip_path = hf_hub_download(
            repo_id="TsinghuaC3I/MedXpertQA",
            filename="images.zip",
            repo_type="dataset",
        )
        print(f"Extracting images to {images_dir}...")
        os.makedirs(images_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(images_dir)
        print("Extraction complete.")

    # Load the MM (multimodal) subset - test split has 2005 samples
    print("Loading MedXpertQA MM subset (test split)...")
    ds = hf_load_dataset(
        "TsinghuaC3I/MedXpertQA",
        name="MM",
        split="test",
        trust_remote_code=True,
    )

    all_rows = []
    skipped = 0

    for item in ds:
        # Get images list - skip if no images
        images = item.get('images', [])
        if not images or len(images) == 0:
            skipped += 1
            continue

        # Use first image (some questions have multiple)
        image_filename = images[0]
        # Images are extracted to medxpertqa_images/images/ subdirectory
        image_path = os.path.join("medxpertqa_images", "images", image_filename)
        full_image_path = os.path.join(cache_dir, image_path)

        # Verify image exists
        if not os.path.exists(full_image_path):
            # Try without the extra 'images' subdirectory
            image_path_alt = os.path.join("medxpertqa_images", image_filename)
            if os.path.exists(os.path.join(cache_dir, image_path_alt)):
                image_path = image_path_alt
            else:
                skipped += 1
                continue

        # Get options dict
        options = item.get('options', {})
        if not options or len(options) < 5:
            skipped += 1
            continue

        # Get correct answer letter and map to text
        label = item.get('label', '').strip().upper()
        if label not in options:
            skipped += 1
            continue

        gt_answer = options[label]

        # Extract question text (remove embedded answer choices if present)
        question_text = item.get('question', '')
        # Remove "Answer Choices: ..." suffix if present
        question_text = re.sub(r'\s*Answer Choices:.*$', '', question_text, flags=re.IGNORECASE | re.DOTALL)
        question_text = question_text.strip()

        row = {
            'question': question_text,
            'gt_answer': gt_answer,
            'image_path': image_path,
            'option_A': options.get('A', ''),
            'option_B': options.get('B', ''),
            'option_C': options.get('C', ''),
            'option_D': options.get('D', ''),
            'option_E': options.get('E', ''),
            'question_type': item.get('medical_task', 'MedXpertQA'),
        }
        all_rows.append(row)

    df = pd.DataFrame(all_rows)
    print(f"Loaded {len(df)} samples from MedXpertQA MM subset (skipped {skipped})")
    return df


def create_omnimed_iid_test_dataset(
    test_csv_path: str = "omnimed_test_split.csv",
) -> tuple[pd.DataFrame, str]:
    """Load the near-IID OmniMedVQA holdout test set.

    Reads question_ids from the test CSV, downloads the full OmniMedVQA
    dataset if needed, and filters to only the matching samples.

    Args:
        test_csv_path: Path to the near-IID test set CSV containing question_id column.

    Returns:
        tuple of (DataFrame, base_path) where DataFrame has the standard schema
        and base_path points to the OmniMedVQA images directory.
    """
    from utils import load_dataset, create_dataset

    if not os.path.exists(test_csv_path):
        raise FileNotFoundError(f"IID test set CSV not found at: {test_csv_path}")

    test_df = pd.read_csv(test_csv_path)
    question_id_filter = set(test_df['question_id'].tolist())
    load_dataset()
    dataset = create_dataset(question_id_filter=question_id_filter)
    base_path = os.path.join("./OmniMedVQA_data", "OmniMedVQA")
    return dataset, base_path


