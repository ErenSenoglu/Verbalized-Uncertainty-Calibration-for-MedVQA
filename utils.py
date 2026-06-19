import os
import pandas as pd
import zipfile
import re
from typing import List, Dict, Optional, Tuple, Union
import numpy as np
from sklearn.metrics import roc_auc_score

EXTRACT_DIR = "./OmniMedVQA_data" 

# Public API of this module
__all__ = [
    "load_dataset",
    "create_dataset",
    "parse_single_response",
    "compute_brier",
    "compute_ece",
    "compute_avg_two_bin_calibration_error",
    "compute_auroc",
    "plot_ece_diagrams",
    "plot_ece_diagrams_similarity",
    "plot_confidence_distribution",
    "plot_gain_distribution",
    "load_method_run0",
    "load_method_agg",
    "create_pmcvqa_train_dataset"
]

def load_dataset(relative_path: str = "."):
    """Download and extract the OmniMedVQA dataset if not already extracted.

    Attempts to download the dataset zip from the Hugging Face Hub and
    extract it into the directory specified by EXTRACT_DIR relative to the
    provided path. If the target directory already exists, the function 
    returns immediately.

    Args:
        relative_path (str): Base directory path where OmniMedVQA_data should
            be located or created. Defaults to current directory.

    Side effects:
        - Creates directories under the target path on disk.
        - Downloads and extracts a zip file if needed.
        - Updates the global EXTRACT_DIR variable.

    Returns:
        None
    """
    global EXTRACT_DIR
    # --- Configuration ---
    repo_id = "foreverbeliever/OmniMedVQA"
    filename = "OmniMedVQA.zip"
    
    # Construct the full path
    EXTRACT_DIR = os.path.join(relative_path, EXTRACT_DIR)
    
    if os.path.exists(EXTRACT_DIR):
        return
    # --- 1. Download the ZIP file from Hugging Face Hub ---
    print(f"Downloading {filename} from {repo_id}...")
    try:
        # Import here to avoid hard dependency when just importing this module
        import importlib
        hf_module = importlib.import_module("huggingface_hub")
        local_zip_path = hf_module.hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type='dataset'
        )
        print(f"Download complete. File saved to: {local_zip_path}")
    except Exception as e:
        print(f"An error occurred during download: {e}")
    
    # --- 2. Extract the ZIP file ---
    print(f"\nExtracting files to {EXTRACT_DIR}...")
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)
    print("Extraction complete.")
    
    # --- 3. Create a black placeholder image ---
    black_path = os.path.join(EXTRACT_DIR, "OmniMedVQA", "black.png")
    if not os.path.exists(black_path):
        # Find any existing image to get dimensions, or use a default size
        try:
            img_size = (520, 520)  # Default size if no images found
            from PIL import Image
            black_img = Image.new("RGB", img_size, (0, 0, 0))
            black_img.save(black_path)
            print(f"Created black placeholder image at: {black_path}")
        except Exception as e:
            print(f"Warning: Could not create black image: {e}")

def create_dataset(question_id_filter: set = None):
    """Create a concatenated QA dataset DataFrame from OmniMedVQA JSON files.

    Reads JSON files under EXTRACT_DIR/OmniMedVQA/QA_information/Open-access,
    concatenates them into a single DataFrame, drops the 'modality' column if
    present.
    
    Args:
        question_id_filter: Optional set of question_id values to filter to.
                            If None, applies default filtering (10 per gt_answer).
                            If provided, returns only matching question_ids with no other filtering.

    Returns:
        pandas.DataFrame | None: The concatenated DataFrame if files are found; otherwise None.
    """
    # Import optional deps inside the function to avoid import-time failures
    import pandas as pd
    import glob
    base_path = os.path.join(EXTRACT_DIR, "OmniMedVQA")
    
    # 1. Define the path pattern to find all JSON files
    # The asterisk (*) is a wildcard that matches any characters.
    json_pattern = os.path.join(base_path, "QA_information", "Open-access", "*.json")
    
    # 2. Use glob to find all files matching the pattern
    file_list = sorted(glob.glob(json_pattern))
    
    if not file_list:
        print(f"No JSON files found at the specified path: {json_pattern}")
    else:
        print(f"Found {len(file_list)} JSON files to concatenate.")
        print(file_list)
    
        # 3. Read each JSON file into a DataFrame and store it in a list
        list_of_dataframes = []
        for file in file_list:
            try:
                # It's good practice to handle potential errors, e.g., an empty or malformed file
                df_single_file = pd.read_json(file)
                
                # Add a column to know the source file of each row, which is very useful
                df_single_file['source_file'] = os.path.basename(file)
                
                list_of_dataframes.append(df_single_file)
            except Exception as e:
                print(f"Error reading {file}: {e}")
    
        # 4. Concatenate all DataFrames in the list into a single DataFrame
        if list_of_dataframes:
            full_df = pd.concat(list_of_dataframes, ignore_index=True)
            if "modality" in full_df.columns:
                full_df = full_df.drop("modality", axis=1)
            
            # Apply filtering based on question_id_filter parameter
            if question_id_filter is not None:
                # Filter to only specified question_ids, no other filtering
                full_df = full_df[full_df['question_id'].isin(question_id_filter)]
                print(f"Filtered to {len(full_df)} samples matching question_id_filter")
            else:
                # Default behavior: limit to 10 examples per gt_answer
                full_df = full_df.groupby('gt_answer', group_keys=False).head(10)
            
            return full_df
    # Nothing built; return None
    return None

def create_pmcvqa_dataset(cache_dir: str = "."):
    """Load PMC-VQA test set and convert to OmniMedVQA-compatible format.
    
    PMC-VQA is a medical VQA dataset from PubMed Central articles.
    The Answer field contains the actual answer text, not a letter.
    
    Args:
        cache_dir: Directory for caching images. Images will be at cache_dir/images/
        
    Returns:
        pandas.DataFrame with columns: question, gt_answer, image_path, 
        option_A, option_B, option_C, option_D, question_type
    """
    import pandas as pd
    from datasets import load_dataset as hf_load_dataset
    
    os.makedirs(cache_dir, exist_ok=True)
    images_dir = os.path.join(cache_dir, "images")
    
    # Download and extract images if not already done
    if not os.path.exists(images_dir):
        print("Downloading PMC-VQA images...")
        from huggingface_hub import hf_hub_download
        zip_path = hf_hub_download(
            repo_id="RadGenome/PMC-VQA",
            filename="images.zip",
            repo_type="dataset",
        )
        print(f"Extracting images to {cache_dir}...")
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(cache_dir)
        print("Extraction complete.")
    
    # Load the test CSV
    print("Loading PMC-VQA test set...")
    ds = hf_load_dataset(
        "csv",
        data_files="hf://datasets/RadGenome/PMC-VQA/test_clean.csv",
        split="train",
    )
    
    def strip_option_prefix(text):
        """Strip the letter prefix like ' A: ' from option text."""
        text = text.strip()
        # Match patterns like "A:", "B:", " A:", " B:" at the start
        return re.sub(r'^[A-D]:\s*', '', text).strip()
    
    all_rows = []
    for item in ds:
        # PMC-VQA Answer field contains the actual answer text, not a letter
        answer_text = item['Answer'].strip()
        
        # Strip prefixes from options
        options = {
            'A': strip_option_prefix(item['Choice A']),
            'B': strip_option_prefix(item['Choice B']),
            'C': strip_option_prefix(item['Choice C']),
            'D': strip_option_prefix(item['Choice D']),
        }
        
        # The gt_answer is the answer text directly
        gt_answer = answer_text
        
        row = {
            'question': item['Question'].strip(),
            'gt_answer': gt_answer,
            'image_path': os.path.join("images", item['Figure_path']),
            'option_A': options['A'],
            'option_B': options['B'],
            'option_C': options['C'],
            'option_D': options['D'],
            'question_type': 'PMC-VQA',
        }
        all_rows.append(row)
    
    df = pd.DataFrame(all_rows)
    print(f"Loaded {len(df)} samples from PMC-VQA test set")
    return df

def create_pmcvqa_full_train_dataset(cache_dir: str = "."):
    """Load full PMC-VQA train set (no sampling) in OmniMedVQA-compatible format.

    Args:
        cache_dir: Directory for caching images. Images will be at cache_dir/images/

    Returns:
        pandas.DataFrame with all valid train samples.
    """
    import pandas as pd
    from datasets import load_dataset as hf_load_dataset

    os.makedirs(cache_dir, exist_ok=True)
    images_dir = os.path.join(cache_dir, "images")

    if not os.path.exists(images_dir):
        print("Downloading PMC-VQA images...")
        from huggingface_hub import hf_hub_download
        zip_path = hf_hub_download(
            repo_id="RadGenome/PMC-VQA",
            filename="images.zip",
            repo_type="dataset",
        )
        print(f"Extracting images to {cache_dir}...")
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(cache_dir)
        print("Extraction complete.")

    print("Loading full PMC-VQA train set...")
    ds = hf_load_dataset(
        "csv",
        data_files="hf://datasets/RadGenome/PMC-VQA/train.csv",
        split="train",
    )
    ds = ds.filter(lambda x: x['Answer'] is not None and x['Question'] is not None)
    print(f"Full train set: {len(ds)} valid rows")

    def strip_option_prefix(text):
        if text is None:
            return ""
        text = text.strip()
        return re.sub(r'^[A-D]:\s*', '', text).strip()

    all_rows = []
    for item in ds:
        answer_text = item['Answer'].strip()
        options = {
            'A': strip_option_prefix(item['Choice A']),
            'B': strip_option_prefix(item['Choice B']),
            'C': strip_option_prefix(item['Choice C']),
            'D': strip_option_prefix(item['Choice D']),
        }
        row = {
            'question': item['Question'].strip(),
            'gt_answer': answer_text,
            'image_path': os.path.join("images", item['Figure_path']),
            'option_A': options['A'],
            'option_B': options['B'],
            'option_C': options['C'],
            'option_D': options['D'],
            'question_type': 'PMC-VQA',
        }
        all_rows.append(row)

    df = pd.DataFrame(all_rows)
    print(f"Loaded {len(df)} samples from full PMC-VQA train set")
    return df


def create_pmcvqa_train_dataset(cache_dir: str = ".", seed: int = 3407):
    """Load PMC-VQA train set, sample 8k rows, and convert to OmniMedVQA-compatible format.

    Args:
        cache_dir: Directory for caching images. Images will be at cache_dir/images/
        seed: Random seed for reproducible sampling (default: 3407)

    Returns:
        pandas.DataFrame with 8000 samples, columns: question, gt_answer, image_path,
        option_A, option_B, option_C, option_D, question_type
    """
    import pandas as pd
    from datasets import load_dataset as hf_load_dataset

    os.makedirs(cache_dir, exist_ok=True)
    images_dir = os.path.join(cache_dir, "images")

    # Download and extract images if not already done
    if not os.path.exists(images_dir):
        print("Downloading PMC-VQA images...")
        from huggingface_hub import hf_hub_download
        zip_path = hf_hub_download(
            repo_id="RadGenome/PMC-VQA",
            filename="images.zip",
            repo_type="dataset",
        )
        print(f"Extracting images to {cache_dir}...")
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(cache_dir)
        print("Extraction complete.")

    # Load the train CSV
    print("Loading PMC-VQA train set...")
    ds = hf_load_dataset(
        "csv",
        data_files="hf://datasets/RadGenome/PMC-VQA/train.csv",
        split="train",
    )

    # Filter out rows with None values in required fields
    ds = ds.filter(lambda x: x['Answer'] is not None and x['Question'] is not None)
    print(f"Filtered to {len(ds)} valid rows")

    # Sample 8000 rows with fixed seed
    ds = ds.shuffle(seed=seed).select(range(8000))
    print(f"Sampled 8000 rows from train set (seed={seed})")

    def strip_option_prefix(text):
        """Strip the letter prefix like ' A: ' from option text."""
        if text is None:
            return ""
        text = text.strip()
        return re.sub(r'^[A-D]:\s*', '', text).strip()

    all_rows = []
    for item in ds:
        answer_text = item['Answer'].strip()

        options = {
            'A': strip_option_prefix(item['Choice A']),
            'B': strip_option_prefix(item['Choice B']),
            'C': strip_option_prefix(item['Choice C']),
            'D': strip_option_prefix(item['Choice D']),
        }

        gt_answer = answer_text

        row = {
            'question': item['Question'].strip(),
            'gt_answer': gt_answer,
            'image_path': os.path.join("images", item['Figure_path']),
            'option_A': options['A'],
            'option_B': options['B'],
            'option_C': options['C'],
            'option_D': options['D'],
            'question_type': 'PMC-VQA',
        }
        all_rows.append(row)

    df = pd.DataFrame(all_rows)
    print(f"Loaded {len(df)} samples from PMC-VQA train set (8k sampled)")
    return df

# Base labeled patterns
answer_pattern = re.compile(r"^\s*Answer:\s*([A-Ea-e])\s*[—\-:]\s*(.+?)\s*$", re.IGNORECASE)
rationale_pattern = re.compile(r"^\s*Rationale:\s*(.+?)\s*$", re.IGNORECASE)
confidence_pattern = re.compile(r"^\s*Confidence:\s*([0-9]+(?:\.[0-9]+)?)\s*$", re.IGNORECASE)
# Flexible 'Answer:' catcher (handles repeated letters and optional punctuation)
answer_flexible_after_prefix = re.compile(
    r"^\s*Answer:\s*"           # prefix
    r"([A-Ea-e])\s*"             # first letter
    r"(?:[:—–\-]?\s*)?"         # optional punct
    r"([A-Ea-e])?\s*"            # optional repeated letter
    r"(?:[:—–\-]?\s*)?"         # optional punct
    r"(.*\S)?\s*$",              # remainder text
    re.IGNORECASE,
)


def _strip_leading_letter_prefix(ans_letter: Optional[str], ans_text: Optional[str]) -> Optional[str]:
    """Remove a duplicated leading answer letter and punctuation from text.

    This is a helper to clean cases like "C : C text" or "B - B something" by
    stripping the redundant letter and optional punctuation at the start of the
    following text.

    Args:
        ans_letter (Optional[str]): The answer letter (A-D) detected.
        ans_text (Optional[str]): The answer text to clean.

    Returns:
        Optional[str]: Cleaned answer text or None if input text is None.
    """
    if not ans_text or not ans_letter:
        return ans_text
    # Remove a leading duplicated letter + optional punct and spaces
    rep = re.compile(rf"^({re.escape(ans_letter)})\b[:\-—\s]*", re.IGNORECASE)
    cleaned = rep.sub("", ans_text).strip()
    # Also handle generic leading pattern like "X " (no punct)
    cleaned = re.sub(rf"^{re.escape(ans_letter)}\s+", "", cleaned, flags=re.IGNORECASE).strip()
    # Strip stray leading punctuation
    cleaned = cleaned.lstrip("—–-: ").strip()
    return cleaned


import numpy as np

def bradley_terry_mm(W, max_iter=2000, tol=1e-9, reg=0.0, init=None):
    """
    Bradley–Terry MLE via MM updates from a pairwise wins matrix.

    Args:
        W: (n x n) array, W[i,j] = # times i beat j. Diagonal must be 0.
        max_iter: max iterations for MM.
        tol: convergence tolerance on L∞ change of probs.
        reg: tiny nonnegative smoothing added to wins/denoms (e.g., 1e-6) to stabilize disconnected cases.
        init: optional initial strengths (length-n, positive).

    Returns:
        P: normalized probs over options (sum=1).
        pred_idx: argmax index (chosen answer).
        pred_conf: P[pred_idx] (confidence).
        iters: number of iterations actually run.
        ll: log-likelihood at the solution.
    """
    W = np.asarray(W, dtype=float)
    assert W.shape[0] == W.shape[1], "W must be square"
    n = W.shape[0]
    assert np.allclose(np.diag(W), 0), "Diagonal of W must be 0"

    # Total matches per pair
    N = W + W.T                      # n_ij = w_ij + w_ji
    wins = W.sum(axis=1)             # total wins for each option

    # Initialize positive strengths (scale-invariant)
    if init is None:
        pi = np.ones(n) / n
    else:
        pi = np.maximum(np.asarray(init, dtype=float), 1e-12)
        pi = pi / pi.sum()

    for it in range(1, max_iter + 1):
        denom = np.zeros(n)
        # denom_i = sum_j n_ij / (pi_i + pi_j)
        for i in range(n):
            s = 0.0
            for j in range(n):
                if i == j: 
                    continue
                nij = N[i, j]
                if nij > 0:
                    s += nij / (pi[i] + pi[j])
            denom[i] = s

        new_pi = (wins + reg) / np.maximum(denom + reg, 1e-12)

        # Remove scale invariance & keep strictly positive
        new_pi = np.maximum(new_pi, 1e-12)
        new_pi = new_pi / new_pi.sum()

        if np.max(np.abs(new_pi - pi)) < tol:
            pi = new_pi
            break
        pi = new_pi

    # Normalize to a categorical distribution
    P = pi / pi.sum()
    pred_idx = int(np.argmax(P))
    pred_conf = float(P[pred_idx])

    return P, pred_idx, pred_conf, it


def pair_rank_from_matrices(pairwise_matrices, **kwargs):
    """
    Apply Bradley–Terry (MM) independently to each question's matrix.

    Args:
        pairwise_matrices: list/array of (n x n) wins matrices.
        **kwargs: forwarded to bradley_terry_mm (max_iter, tol, reg, init).

    Returns:
        results: list of dicts with probs, pred index, confidence, iters per question.
    """
    results = []
    for W in pairwise_matrices:
        P, pred_idx, pred_conf, iters = bradley_terry_mm(W, **kwargs)
        results.append({
            "pred_index": pred_idx,
            "confidence": pred_conf
        })
    return results




def parse_single_response(text: str) -> Dict[str, Optional[Union[str, float]]]:
    """Parse a model response to extract rationale, answer, and confidence.

    Supports both labeled multi-line format and several condensed formats, e.g.:
      - "C : C — There are no specific abnormalities observed in this image."
      - "B : B Cuneonavicular Articulation"
      - "B — COVID-19 negative" / "B - COVID-19 negative"
      - "B : bowel enlargement"
      - "Answer: C C COVID-19 negative" (repeated letter with or without punctuation)

    Args:
        text (str): The raw text response produced by the model.

    Returns:
        Dict[str, Optional[Union[str, float]]]: A dictionary with keys:
            - rationale: Optional[str]
            - answer_letter: Optional[str]  (A-D)
            - answer_text: Optional[str]
            - confidence: Optional[float]
    """
    rationale = None
    answer_letter: Optional[str] = None
    answer_text: Optional[str] = None
    confidence: Optional[str] = None

    s_text = str(text)

    # Pass 1: try to parse labeled multi-line format
    for line in s_text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = rationale_pattern.match(line)
        if m and rationale is None:
            rationale = m.group(1).strip()
            continue

        # Try strict labeled answer first
        m = answer_pattern.match(line)
        if m and answer_letter is None:
            answer_letter = m.group(1).upper()
            answer_text = m.group(2).strip()
            answer_text = re.sub(r'^[A-Ea-e]\s*[:\-—]\s*', '', answer_text).strip()
            continue

        # Then a flexible 'Answer:' handler to catch variants like "Answer: C C text" or "Answer: C text"
        if line.lower().startswith("answer:") and answer_letter is None:
            m2 = answer_flexible_after_prefix.match(line)
            if m2:
                first = (m2.group(1) or '').upper()
                second = (m2.group(2) or '').upper()
                rest = (m2.group(3) or '').strip()
                # Choose the more informative letter: prefer the second if present, else first
                answer_letter = (second or first) or None
                answer_text = rest
                # Clean possible repeated leading letter inside the text
                answer_text = _strip_leading_letter_prefix(answer_letter, answer_text)
                continue

        m = confidence_pattern.match(line)
        if m and confidence is None:
            confidence = m.group(1).strip()
            continue

    # Pass 2: handle condensed formats if not resolved yet
    if answer_letter is None and (answer_text is None or answer_text == ""):
        lines = [ln.strip() for ln in s_text.splitlines() if ln.strip()] or [s_text.strip()]

        # Patterns for condensed lines (no 'Answer:' prefix)
        rx_double_letter = re.compile(r"^([A-Ea-e])\s*:\s*([A-Ea-e])\s*(?:[—–\-]\s*)?(.*\S)?\s*$")
        rx_colon_text = re.compile(r"^([A-Ea-e])\s*:\s*(.*\S)\s*$")
        rx_dash_text = re.compile(r"^([A-Ea-e])\s*(?:[—–\-])\s*(.*\S)\s*$")
        rx_space_text = re.compile(r"^([A-Ea-e])\s+(.*\S)\s*$")  # handles "C COVID-19 negative"

        def _clean_ans(t: Optional[str]) -> Optional[str]:
            if t is None:
                return None
            t = t.lstrip("—–-: ").strip()
            return t if t != "" else None

        matched = False
        for ln in lines:
            m = rx_double_letter.match(ln)
            if m:
                answer_letter = m.group(2).upper()
                answer_text = _clean_ans(m.group(3))
                matched = True
                break
        if not matched:
            for ln in lines:
                m = rx_colon_text.match(ln)
                if m:
                    answer_letter = m.group(1).upper()
                    answer_text = _clean_ans(m.group(2))
                    matched = True
                    break
        if not matched:
            for ln in lines:
                m = rx_dash_text.match(ln)
                if m:
                    answer_letter = m.group(1).upper()
                    answer_text = _clean_ans(m.group(2))
                    matched = True
                    break
        if not matched:
            for ln in lines:
                m = rx_space_text.match(ln)
                if m:
                    answer_letter = m.group(1).upper()
                    answer_text = _clean_ans(m.group(2))
                    matched = True
                    break

        # Final cleanup if text still begins with the repeated letter/punct
        if answer_text and answer_letter:
            answer_text = _strip_leading_letter_prefix(answer_letter, answer_text)

    # Simple final fallback: if answer_text literally starts with "LETTER " then strip it
    if answer_letter and isinstance(answer_text, str):
        prefix = f"{answer_letter} "
        if answer_text.upper().startswith(prefix.upper()):
            answer_text = answer_text[len(prefix):].lstrip("—–-: ").strip()

    return {
        "rationale": rationale,
        "answer_letter": answer_letter,
        "answer_text": answer_text,
        "confidence": float(confidence) if confidence is not None else None,
    }


def _norm_text(x):
    """Normalize text for robust equality checks.

    Lowercases, trims whitespace, normalizes dashes to '-', collapses internal
    spaces, and removes a trailing period if present.

    Args:
        x: Any input; will be converted to string if not None.

    Returns:
        Optional[str]: Normalized text, or None if x is None.
    """
    if x is None:
        return None
    if not isinstance(x, str):
        x = str(x)
    x = x.strip().lower()
    # Strip PMC-VQA style option prefixes like "a:", "b:", " a: ", etc.
    x = re.sub(r"^[a-d]:\s*", "", x)
    x = re.sub(r"[\u2013\u2014\u2012\-]+", "-", x)
    # Strip trailing option letter suffix like " - a", " - b", etc.
    x = re.sub(r"\s*-\s*[a-e]$", "", x)
    x = re.sub(r"\s+", " ", x)
    if len(x) > 0:
        if x[-1] == ".":
            x = x[:-1]
    return x

# Helper to parse run 0 for a given method/temperature
def load_method_run0(method_name: str, temp: float, num_runs_candidates=(10, 5), relative_path: str ="") -> Tuple[Optional['pd.DataFrame'], Optional[int], Optional[np.ndarray], Optional[np.ndarray]]:
    """Load run-0 generations for a method/temperature and parse responses.

    Loads the "*_all_generations_*runs_temp{temp}.npy" array and the corresponding
    ground-truths array. Parses the run-0 generations into a DataFrame with
    extracted fields and correctness flags. Also augments the DataFrame with
    length-normalized score for run 0 and its derived perplexity if available
    via files named "{method_name}_length_normalized_scores_{runs}runs_temp{temp}.npy".

    Args:
        method_name (str): Prefix of file names (e.g., "black_image" or "verbalized").
        temp (float): Temperature used when generating outputs.
        num_runs_candidates (tuple[int, int]): Candidate counts of runs to try (in order).

    Returns:

        tuple: (df0, picked, gens, gts) where
            - df0 (pandas.DataFrame | None): Parsed run-0 DataFrame or None on failure.
                - df0 columns: 'rationale', 'answer_letter', 'answer_text', 'confidence', 'sample_idx',
                    'gt_answer', 'answer_text_norm', 'gt_answer_norm', 'correct',
                    'length_norm_score', 'perplexity' (if available)
            - picked (int | None): The number of runs picked based on available files.
            - gens (numpy.ndarray | None): All generations array.
            - gts (numpy.ndarray | None): Ground-truth answers array.

    """
    # Import optional dependency locally
    import pandas as pd
    temp_str = str(temp).replace('.', '_')
    picked = None
    gens = None
    for nr in num_runs_candidates:
        p = f"{relative_path}{method_name}_all_generations_{nr}runs_temp{temp_str}.npy"
        if os.path.exists(p):
            gens = np.load(p, allow_pickle=True)
            picked = nr
            lns = []
            path = f"{relative_path}{method_name}_all_logprobs_{nr}runs_temp{temp_str}.npy"
            if os.path.exists(path):
                ln_scores = np.load(path, allow_pickle=True)
                tok = f"{relative_path}{method_name}_all_decoded_tokens_{nr}runs_temp{temp_str}.npy"
                if os.path.exists(tok):
                    decoded = np.load(tok, allow_pickle=True)
                    for i in range(decoded.shape[0]):
                        toks = decoded[i][0]
                        idx = next((j for j, t in enumerate(toks) if t == 'Confidence' or str(t).lower() == 'confidence'), None)
                        if idx is not None:
                            ln_scores[i][0] = list(ln_scores[i][0])[:idx]
                for item in ln_scores:
                    lns.append(np.exp(np.array(item[0]).mean()))
            break
        else:
            continue    

        
    if gens is None:
        print(f"[WARN] Missing generations for {method_name} @ T={temp}")
        return None, None, None, None

    gt_path = f"{relative_path}{method_name}_ground_truths.npy"
    if not os.path.exists(gt_path):
        print(f"[WARN] Missing ground truths for {method_name}")
        return None, None, None, None
    gts = np.load(gt_path, allow_pickle=True)
    
    

    # Align lengths
    num_samples = gens.shape[0]
    if len(gts) != num_samples:
        m = min(len(gts), num_samples)
        gens = gens[:m, :]
        gts = gts[:m]
        num_samples = m


    # Parse run 0
    rows = []
    for i in range(num_samples):
        rec = parse_single_response(gens[i, 0])
        rec.update({"sample_idx": i})
        rows.append(rec)
    df0 = pd.DataFrame(rows)
    df0['gt_answer'] = gts
    df0['perplexity'] = lns if lns else pd.NA

    # Correctness via text match (shuffle-safe)
    df0['answer_text_norm'] = df0['answer_text'].apply(_norm_text)
    df0['gt_answer_norm']   = df0['gt_answer'].apply(_norm_text)
    df0['correct'] = (df0['answer_text_norm'] == df0['gt_answer_norm'])

    #replace nan values in confidence column with mean and then report the number of replacements
    num_replaced = df0['confidence'].isna().sum()
    if num_replaced > 0:
        mean_conf = df0['confidence'].mean()
        mean_conf = int(mean_conf)
        df0['confidence'] = df0['confidence'].fillna(mean_conf)
        print(f"[INFO] Replaced {num_replaced} NaN confidence values with mean: {mean_conf}")

    return df0, picked, gens, gts


def load_method_run10(method_name: str, temp: float, num_runs_candidates=(10, 5), relative_path="") -> Tuple[Optional['pd.DataFrame'], Optional[int], Optional[np.ndarray], Optional[np.ndarray]]:
    import pandas as pd

    temp_str = str(temp).replace('.', '_')
    nr = num_runs_candidates[0]

    p = f"{relative_path}{method_name}_all_generations_{nr}runs_temp{temp_str}.npy"
    if os.path.exists(p):
        gens = np.load(p, allow_pickle=True)
        picked = nr
        lns = []
        path = f"{relative_path}{method_name}_all_logprobs_{nr}runs_temp{temp_str}.npy"
        if os.path.exists(path):
            ln_scores = np.load(path, allow_pickle=True)
            tok = f"{relative_path}{method_name}_all_decoded_tokens_{nr}runs_temp{temp_str}.npy"
            if os.path.exists(tok):
                decoded = np.load(tok, allow_pickle=True)
                for i in range(decoded.shape[0]):
                    for z in range(nr):
                        toks = decoded[i][z]
                        idx = next((j for j, t in enumerate(toks) if t == 'Confidence' or str(t).lower() == 'confidence'), None)
                        if idx is not None:
                            ln_scores[i][z] = list(ln_scores[i][z])[:idx]
                for item in ln_scores:
                    for instance in item:
                        lns.append(np.exp(np.array(item[0]).mean()))

    gt_path = f"{relative_path}{method_name}_ground_truths.npy"
    gts = np.load(gt_path, allow_pickle=True)
    
    num_samples = gens.shape[0]
    # Parse run 0
    rows = []
    for i in range(num_samples):
        for j in range(nr):
            rec = parse_single_response(gens[i, j])
            rec.update({"sample_idx": i})
            rec.update({"run_idx": j})
            rows.append(rec)

    df0 = pd.DataFrame(rows)
    df0['gt_answer'] = np.repeat(gts, nr)
    df0['perplexity'] = lns if lns else pd.NA

    # Correctness via text match (shuffle-safe)
    df0['answer_text_norm'] = df0['answer_text'].apply(_norm_text)
    df0['gt_answer_norm']   = df0['gt_answer'].apply(_norm_text)
    df0['correct'] = (df0['answer_text_norm'] == df0['gt_answer_norm'])

    #replace nan values in confidence column with mean and then report the number of replacements
    num_replaced = df0['confidence'].isna().sum()
    if num_replaced > 0:
        mean_conf = df0['confidence'].mean()
        mean_conf = int(mean_conf)
        df0['confidence'] = df0['confidence'].fillna(mean_conf)
        print(f"[INFO] Replaced {num_replaced} NaN confidence values with mean: {mean_conf}")

    
    df0['accuracy'] = df0.groupby(df0.index // nr)['correct'].transform('mean')
    return df0

    

def load_method_agg(method_name: str, temp: float, num_runs_candidates=(10, 5), k=10) -> Tuple[Optional['pd.DataFrame'], Optional[int], Optional[np.ndarray], Optional[np.ndarray]]:
    """Load agg generations for a method/temperature and parse responses.

    Loads the "*_all_generations_*runs_temp{temp}.npy" array and the corresponding
    ground-truths array. Parses the run-0 generations into a DataFrame with
    extracted fields and correctness flags. Also augments the DataFrame with
    length-normalized score for run 0 and its derived perplexity if available
    via files named "{method_name}_length_normalized_scores_{runs}runs_temp{temp}.npy".

    Args:
        method_name (str): Prefix of file names (e.g., "black_image" or "verbalized").
        temp (float): Temperature used when generating outputs.
        num_runs_candidates (tuple[int, int]): Candidate counts of runs to try (in order).

    Returns:
        tuple: (df0, picked, gens, gts) where
            - df0 (pandas.DataFrame | None): Parsed run-0 DataFrame or None on failure.
                - df0 columns: 'rationale', 'answer_letter', 'answer_text', 'confidence', 'sample_idx',
                    'gt_answer', 'answer_text_norm', 'gt_answer_norm', 'correct',
                    'length_norm_score', 'perplexity' (if available)
            - picked (int | None): The number of runs picked based on available files.
            - gens (numpy.ndarray | None): All generations array.
            - gts (numpy.ndarray | None): Ground-truth answers array.

    """
    # Import optional dependency locally
    import pandas as pd
    temp_str = str(temp).replace('.', '_')
    picked = None
    gens = None
    for nr in num_runs_candidates:
        p = f"{method_name}_all_generations_{nr}runs_temp{temp_str}.npy"
        if os.path.exists(p):
            gens = np.load(p, allow_pickle=True)
            picked = nr
            lns = []
            path = f"{method_name}_all_logprobs_{nr}runs_temp{temp_str}.npy"
            if os.path.exists(path):
                ln_scores = np.load(path, allow_pickle=True) if os.path.exists(path) else None
                for item in ln_scores:
                    lns.append(np.exp(np.array(item[0]).mean()))
        break

        
    if gens is None:
        print(f"[WARN] Missing generations for {method_name} @ T={temp}")
        return None, None, None, None

    gt_path = f"{method_name}_ground_truths.npy"
    if not os.path.exists(gt_path):
        print(f"[WARN] Missing ground truths for {method_name}")
        return None, None, None, None
    gts = np.load(gt_path, allow_pickle=True)
    
    

    # Align lengths
    num_samples = gens.shape[0]
    if len(gts) != num_samples:
        m = min(len(gts), num_samples)
        gens = gens[:m, :]
        gts = gts[:m]
        num_samples = m


    # Parse aggregated runs
    rows = []
    for i in range(gens.shape[0]):
        conf = 0
        freq = []
        rec = {}
        for j in range((k-1), -1, -1):
            rec = parse_single_response(gens[i][j])
            conf += rec["confidence"] if rec["confidence"] is not None else 0
            freq.append(rec["answer_text"])
        conf /= k
        rec.update({"confidence": conf})
        rec.update({"sample_idx": i})
        rec.update({"freq": freq})
        rows.append(rec)

    df0 = pd.DataFrame(rows)
    df0['gt_answer'] = gts
    df0['perplexity'] = lns if lns else pd.NA

    # Correctness via text match (shuffle-safe)
    df0['answer_text_norm'] = df0['answer_text'].apply(_norm_text)
    df0['gt_answer_norm']   = df0['gt_answer'].apply(_norm_text)
    df0['correct'] = (df0['answer_text_norm'] == df0['gt_answer_norm'])
    
    # Normalize frequency answers and compute consistency score
    df0['freq_norm'] = df0['freq'].apply(lambda x: [_norm_text(ans) for ans in x])
    
    # For each row, count how many other runs match run 0's prediction (leave-one-out)
    df0['consistency'] = df0.apply(lambda row: 
        sum(1 for ans in row['freq_norm'][1:] if ans == row['freq_norm'][0]) / (len(row['freq_norm'])-1), 
        axis=1
    )

    df0.drop(columns=['freq_norm', 'freq'], inplace=True)

    return df0, picked, gens, gts


def _extract_two_answers_with_confidence(text: str):
    answer_letter_first = re.compile(r"^\s*Guess:\s*([A-Ea-e])\s*[—\-:]\s*(.+?)\s*$", re.IGNORECASE)
    answer_text_first  = re.compile(r"^\s*Guess:\s*(.+?)\s*[—\-:]\s*([A-Ea-e])\s*$", re.IGNORECASE)
    lines = [ln.strip() for ln in str(text).splitlines() if ln.strip()]
    pairs = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m1 = answer_letter_first.match(line)
        m2 = answer_text_first.match(line)
        if m1 or m2:
            if m1:
                ans_letter = m1.group(1).upper()
                ans_text = (m1.group(2) or '').strip()
                # Clean repeated leading letter if present
                try:
                    ans_text = _strip_leading_letter_prefix(ans_letter, ans_text)
                except NameError:
                    pass
            else:
                ans_text = (m2.group(1) or '').strip()
                ans_letter = (m2.group(2) or '').upper()
            # Find the next Confidence line after this Answer
            conf_val = None
            j = i + 1
            while j < len(lines):
                cm = confidence_pattern.match(lines[j])
                if cm:
                    conf_val = cm.group(1).strip()
                    break
                j += 1
            pairs.append({
                'rationale': None,
                'answer_letter': ans_letter,
                'answer_text': ans_text,
                'confidence': float(conf_val) if conf_val is not None else None,
            })
            # Skip to after the confidence if we found one; otherwise advance one
            i = (j + 1) if (conf_val is not None) else (i + 1)
            if len(pairs) == 2:
                break
            continue
        i += 1
    # Ensure two entries (pad with Nones if missing)
    while len(pairs) < 2:
        pairs.append({'rationale': None, 'answer_letter': None, 'answer_text': None, 'confidence': None})
    return pairs[0], pairs[1]


def load_method_topk(method_name: str, temp: float, k: int, num_runs_candidates=(10, 5)) -> Tuple[Optional['pd.DataFrame'], Optional[int], Optional[np.ndarray], Optional[np.ndarray]]:
    """Load top-k generations for a method/temperature and parse responses.

    Loads the "*_all_generations_*runs_temp{temp}.npy" array and the corresponding
    ground-truths array. Parses the top-k generations into a DataFrame with
    extracted fields and correctness flags.

    Args:
        method_name (str): Prefix of file names (e.g., "black_image" or "verbalized").
        temp (float): Temperature used when generating outputs.
        k (int): Number of top generations to consider.
        num_runs_candidates (tuple[int, int]): Candidate counts of runs to try (in order).
    
    Returns:
        tuple: (df0, picked, gens, gts) where
            - df0 (pandas.DataFrame | None): Parsed top-k DataFrame or None on failure.
                - df0 columns: 'rationale', 'answer_letter', 'answer_text', 'confidence', 'sample_idx',
                    'gt_answer', 'answer_text_norm', 'gt      _answer_norm', 'correct'
            - picked (int | None): The number of runs picked based on available files.
            - gens (numpy.ndarray | None): All generations array.
            - gts (numpy.ndarray | None): Ground-truth answers array.   
    """

    #Import optional dependency locally
    from collections import defaultdict
    import pandas as pd
    temp_str = str(temp).replace('.', '_')
    picked = None
    gens = None
    for nr in num_runs_candidates:
        p = f"{method_name}_all_generations_{nr}runs_temp{temp_str}.npy"
        if os.path.exists(p):
            gens = np.load(p, allow_pickle=True)
            picked = nr
            lns = []
            path = f"{method_name}_all_logprobs_{nr}runs_temp{temp_str}.npy"
            if os.path.exists(path):
                ln_scores = np.load(path, allow_pickle=True) if os.path.exists(path) else None
                for item in ln_scores:
                    lns.append(np.exp(np.array(item[0]).mean()))
        break

        
    if gens is None:
        print(f"[WARN] Missing generations for {method_name} @ T={temp}")
        return None, None, None, None

    gt_path = f"{method_name}_ground_truths.npy"
    if not os.path.exists(gt_path):
        print(f"[WARN] Missing ground truths for {method_name}")
        return None, None, None, None
    gts = np.load(gt_path, allow_pickle=True)
    
    

    # Align lengths
    num_samples = gens.shape[0]
    if len(gts) != num_samples:
        m = min(len(gts), num_samples)
        gens = gens[:m, :]
        gts = gts[:m]
        num_samples = m

    pairwise_counts = []          # list[defaultdict[str, defaultdict[str, int]]]
    pairwise_matrices = []        # list[np.ndarray]
    answer_lists = []             # list[list[str]] mapping row/col index -> answer text per question
    rationales = []               # list[list[str]] rationales per question (not used here)
    answer_letters = []
    avg_confidences = []          # list[float | None] mean confidence of first guesses per question
    run0_confidences = []         # list[float | None] confidence of first guess per question

    # Determine dimensions
    total_questions = gens.shape[0]
    num_runs = gens.shape[1] if gens.ndim > 1 else 1

    for i in range(total_questions):
        counts = defaultdict(lambda: defaultdict(int))
        answers_set = set()
        first_guess_confs = []

        # Collect all answers and populate ordered pair counts for this question
        for j in range(num_runs):
            text = gens[i, j]
            g1, g2 = _extract_two_answers_with_confidence(text)
            if j == 0:
                rationales.append(g1.get('rationale'))
                answer_letters.append(g1.get('answer_letter'))
                run0_confidences.append(g1.get('confidence'))

            a1 = _norm_text(g1.get('answer_text')) if isinstance(g1, dict) else None
            a2 = _norm_text(g2.get('answer_text')) if isinstance(g2, dict) else None

            if g1.get('confidence') is not None:
                first_guess_confs.append(g1.get('confidence'))

            if a1:
                answers_set.add(a1)
            if a2:
                answers_set.add(a2)
            if a1 and a2 and a1 != a2:
                counts[a1][a2] += 1

        # Compute mean confidence of first guesses for this question
        if len(first_guess_confs) > 0:
            avg_confidences.append(float(np.mean(first_guess_confs)))
        else:
            avg_confidences.append(-1)

        # Build index and matrix for this question
        answers = sorted(answers_set)
        answer_lists.append(answers)
        pairwise_counts.append(counts)

        n = len(answers)
        if n == 0:
            pairwise_matrices.append(np.zeros((0, 0), dtype=int))
            continue

        idx = {ans: k for k, ans in enumerate(answers)}
        mat = np.zeros((n, n), dtype=int)
        for a1 in counts:
            for a2 in counts[a1]:
                i1 = idx.get(a1)
                i2 = idx.get(a2)
                if i1 is None or i2 is None:
                    continue
                mat[i1, i2] = counts[a1][a2]
        pairwise_matrices.append(mat)


    pred_indices, confidence_scores = [], []
    for i, item in enumerate(pairwise_matrices):
        try:
            result = pair_rank_from_matrices([item], reg=1e-9)[0]
            pred_indices.append(result["pred_index"])
            confidence_scores.append(result["confidence"])
        except Exception as e:
            pred_indices.append(None)
            confidence_scores.append(None)
            continue


    df0 = pd.DataFrame()
    df0['rationale'] = rationales
    df0['answer_letter'] = answer_letters
    df0['answer_text'] = [
        answer_lists[i][pred_indices[i]] if pred_indices[i] is not None and pred_indices[i] < len(answer_lists[i]) else None
        for i in range(total_questions)
    ]
    df0['confidence'] = confidence_scores
    df0['avg_confidence'] = avg_confidences
    df0['avg_confidence'] = df0['avg_confidence'].apply(lambda x: None if x == -1 else x/100.0)
    df0['run0_confidence'] = run0_confidences
    df0['run0_confidence'] = df0['run0_confidence'].apply(lambda x: None if x is None else x/100.0)
    df0['sample_idx'] = list(range(total_questions))
    df0['gt_answer'] = gts
    df0['perplexity'] = lns if lns else pd.NA
    # Correctness via text match (shuffle-safe)
    df0['answer_text_norm'] = df0['answer_text'].apply(_norm_text)
    df0['gt_answer_norm']   = df0['gt_answer'].apply(_norm_text)
    df0['correct'] = (df0['answer_text_norm'] == df0['gt_answer_norm'])

    #replace nan values in confidence column with mean and then report the number of replacements
    num_replaced = df0['confidence'].isna().sum()
    if num_replaced > 0:
        mean_conf = df0['confidence'].mean()
        df0['confidence'] = df0['confidence'].fillna(mean_conf)
        print(f"[INFO] Replaced {num_replaced} NaN confidence values with mean: {mean_conf}")

    return df0, picked, gens, gts


# Helper: Brier/ECE on probabilities
def compute_brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute the Brier score for binary outcomes.

    Args:
        y_true (np.ndarray): Binary labels in {0,1}.
        y_prob (np.ndarray): Predicted probabilities in [0,1].

    Returns:
        float: Mean squared error between y_true and y_prob.
    """
    brier = float(np.mean((y_true - y_prob) ** 2))
    return brier

def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    source_scale: int,
    n_bins: int = 10,
) -> Tuple[float, List[int], List[float]]:
    """Compute Expected Calibration Error (ECE) and per-bin stats.

    Args:
        y_true (np.ndarray): Binary labels in {0,1}.
        y_prob (np.ndarray): Predicted probabilities in [0,1].
        n_bins (int): Number of bins used for calibration.

    Returns:
        Tuple[float, List[int], List[float]]: (ece, bin_counts, bin_accuracies)
            - ece: Weighted average absolute difference between accuracy and confidence.
            - bin_counts: Number of samples in each bin.
            - bin_accuracies: Accuracy per bin.
    """
    y_prob = np.asarray(y_prob, float)
    y_true = np.asarray(y_true, float)

    # Quantize once to the data’s true resolution
    if source_scale == 100:
        y_prob = np.round(y_prob, 2)
    elif source_scale == 10:
        y_prob = np.round(y_prob, 1)


    bin_counts = []
    bin_accuracies = [] 
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    N = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        lo = np.round(lo, 1)
        hi = np.round(hi, 1)
        in_bin = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        n_b = np.sum(in_bin)
        if n_b == 0:
            bin_counts.append(0)
            bin_accuracies.append(0)
            continue
        acc_b = np.mean(y_true[in_bin])
        conf_b = np.mean(y_prob[in_bin])
        ece += (n_b / N) * abs(acc_b - conf_b)
        bin_counts.append(n_b)
        bin_accuracies.append(acc_b)

    return float(ece), bin_counts, bin_accuracies


def compute_avg_two_bin_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    source_scale: int,
) -> float:
    """Compute Averaged Two-Bin Calibration Error (ATB) via exact interval sweep.

    This follows the "surgical" ATB algorithm:
      1) bias_i = p_i - y_i
      2) sort by p_i (ascending), carrying bias_i
      3) pad probabilities with 0 and 1
      4) build cumulative left/right bias sums for split points
      5) integrate width * (left_sum^2 + right_sum^2) over [0,1]
      6) normalize by T^2, where T is valid sample count

    Args:
        y_true (np.ndarray): Binary labels in {0,1}.
        y_prob (np.ndarray): Predicted probabilities in [0,1].
        source_scale (int): Original confidence scale hint (10 or 100) used
            for quantization consistency with compute_ece.

    Returns:
        float: Averaged two-bin calibration error.
    """
    y_prob = np.asarray(y_prob, float)
    y_true = np.asarray(y_true, float)

    # Keep quantization behavior consistent with compute_ece.
    if source_scale == 100:
        y_prob = np.round(y_prob, 2)
    elif source_scale == 10:
        y_prob = np.round(y_prob, 1)

    # Drop invalid values.
    valid = np.isfinite(y_prob) & np.isfinite(y_true)
    if not np.any(valid):
        return float("nan")

    y_prob = np.clip(y_prob[valid], 0.0, 1.0)
    y_true = y_true[valid]
    T = int(len(y_true))
    if T == 0:
        return float("nan")

    biases = y_prob - y_true

    # Stable sort by probability and carry corresponding biases.
    order = np.argsort(y_prob, kind="mergesort")
    sorted_p = y_prob[order]
    sorted_biases = biases[order]

    # Pad split boundaries so the sweep covers the full [0, 1] interval.
    p_padded = np.empty(T + 2, dtype=float)
    p_padded[0] = 0.0
    p_padded[1:T + 1] = sorted_p
    p_padded[T + 1] = 1.0

    # left_sums[k] = sum_{i < k} sorted_biases[i], k in [0..T]
    left_sums = np.empty(T + 1, dtype=float)
    left_sums[0] = 0.0
    left_sums[1:] = np.cumsum(sorted_biases)

    total_bias = float(left_sums[-1])
    # right_sums[k] = sum_{i >= k} sorted_biases[i]
    right_sums = total_bias - left_sums

    widths = p_padded[1:] - p_padded[:-1]  # length T+1
    sq_errors = (left_sums ** 2) + (right_sums ** 2)

    atb_score = float(np.sum(widths * sq_errors))
    return float(atb_score / (T * T))

def compute_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC).

    Args:
        y_true (np.ndarray): Binary labels in {0,1}.
        y_prob (np.ndarray): Predicted probabilities/confidences.

    Returns:
        float: AUROC score.
    """
    try:
        return float(roc_auc_score(y_true, y_prob))
    except ValueError:
        return float('nan')


def calculate_steer_conf(df, l=5):
    """Calculate steered confidence from l repeated prompts with different confidence styles.

    Each block of l consecutive samples represents the same question asked with
    different confidence steering prompts. This function aggregates them into a
    single calibrated confidence score per question.
    """
    import pandas as pd
    df['steer_conf_f'] = float('nan')
    answers = []
    for i in range(0, len(df), l):
        block = df.iloc[i:i+l]
        cmin = block['confidence'].min()
        cmax = block['confidence'].max()

        kmean = block['confidence'].mean()
        kstd = block['confidence'].std()
        kconf = 0 if (pd.isna(kmean) or kmean == 0) else 1/(1+(kstd / kmean))
        kans = block['answer_text_norm'].value_counts(normalize=True).max()
        df.loc[df.index[i], 'k_mean'] = kmean
        df.loc[df.index[i], 'k_std'] = kstd
        df.loc[df.index[i], 'k_conf'] = kconf
        df.loc[df.index[i], 'k_ans'] = kans
        steer_conf = kconf * kans * kmean
        df.loc[df.index[i], 'steer_conf'] = steer_conf
        if cmax == cmin:
            mode_result = block['answer_text_norm'].mode()
            if len(mode_result) > 0:
                most_frequent_answer = mode_result.iloc[0]
                j = block[block['answer_text_norm'] == most_frequent_answer].index[0] - block.index[0]
                j = int(j)
            else:
                j = 0
        else:
            j = (steer_conf - cmin) / (cmax - cmin) * (2*l+1)
            j = np.floor(j).astype(int)
            j = j.clip(0, l-1)

        df.loc[df.index[i], 'j'] = j
        df.loc[df.index[i], 'c_max'] = cmax
        df.loc[df.index[i], 'c_min'] = cmin
        answers.append(block.iloc[j]['answer_text_norm'])
    mask = df.sample_idx % l == 0
    df = df[mask].reset_index(drop=True)
    df['steered_answer'] = answers
    df['correct'] = (df['steered_answer'] == df['gt_answer_norm'])
    return df


def calculate_topk(method_name: str, temp: float, num_runs_candidates=(5, 10, 1), relative_path: str = ""):
    """Calculate top-k confidence by aggregating k guesses across N runs.

    For each question, pools all k*N (answer, confidence) pairs from the top-k
    generations, groups by normalized answer text, computes mean confidence per
    answer, and selects the answer with the highest mean confidence.

    Args:
        method_name: Prefix of the npy files.
        temp: Temperature used when generating.
        num_runs_candidates: Candidate run counts to try.
        relative_path: Directory containing the npy files.

    Returns:
        DataFrame with one row per question: answer_text, confidence (0-1 scale),
        gt_answer, correct, n_guesses (how many guess pairs contributed).
    """
    import pandas as pd
    from collections import defaultdict

    temp_str = str(temp).replace('.', '_')
    gens = None
    for nr in num_runs_candidates:
        p = os.path.join(relative_path, f"{method_name}_all_generations_{nr}runs_temp{temp_str}.npy")
        if os.path.exists(p):
            gens = np.load(p, allow_pickle=True)
            break

    if gens is None:
        print(f"[WARN] Missing generations for {method_name} @ T={temp}")
        return None

    gt_path = os.path.join(relative_path, f"{method_name}_ground_truths.npy")
    if not os.path.exists(gt_path):
        print(f"[WARN] Missing ground truths for {method_name}")
        return None
    gts = np.load(gt_path, allow_pickle=True)

    total_questions = gens.shape[0]
    num_runs = gens.shape[1] if gens.ndim > 1 else 1

    results = []
    for i in range(total_questions):
        # Collect all (answer, confidence) pairs across runs and guesses
        answer_confs = defaultdict(list)

        for j in range(num_runs):
            text = gens[i, j] if gens.ndim > 1 else gens[i]
            g1, g2 = _extract_two_answers_with_confidence(text)

            for g in [g1, g2]:
                ans = _norm_text(g.get('answer_text'))
                conf = g.get('confidence')
                if ans and conf is not None:
                    answer_confs[ans].append(conf)

        # Pick answer with highest mean confidence
        if answer_confs:
            best_answer = max(answer_confs, key=lambda a: np.mean(answer_confs[a]))
            best_conf = np.mean(answer_confs[best_answer]) / 100.0  # 0-100 -> 0-1
        else:
            best_answer = None
            best_conf = None

        gt_norm = _norm_text(str(gts[i]))
        results.append({
            'answer_text_norm': best_answer,
            'confidence': best_conf,
            'gt_answer': gts[i],
            'gt_answer_norm': gt_norm,
            'correct': best_answer == gt_norm if best_answer else False,
            'n_guesses': sum(len(v) for v in answer_confs.values()),
            'sample_idx': i,
        })

    df = pd.DataFrame(results)

    # Fill NaN confidences with mean
    num_replaced = df['confidence'].isna().sum()
    if num_replaced > 0:
        mean_conf = df['confidence'].mean()
        df['confidence'] = df['confidence'].fillna(mean_conf)
        print(f"[INFO] Replaced {num_replaced} NaN confidence values with mean: {mean_conf:.4f}")

    return df


import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Dict, Union
def plot_confidence_distribution(
    confidences: np.ndarray,
    correctness: np.ndarray,
    source_scale: int,
    range_max: int = 1,
    ax: Optional['plt.Axes'] = None,
    title: Optional[str] = None,
    y_lim: Optional[Tuple[float, float]] = None,
    n_bins: int = 10,
    confidences2: Optional[np.ndarray] = None,
    correctness2: Optional[np.ndarray] = None,
    label1: str = "Set 1",
    label2: str = "Set 2",
    ):
    """Plot a minimal frequency histogram using the same binning logic as compute_ece.

    Binning strictly follows: edges = np.linspace(0.0, 1.0, n_bins+1) on normalized values.
    The x-axis is shown in the chosen range [0, range_max] with n_bins bars.

    Args:
        confidences: Array-like confidence values (expected in [0, range_max]).
        correctness: Array-like 0/1 or bool of same length (used only for mean accuracy).
        source_scale: Scale of source data (e.g. 100 for percentages).
        range_max: One of {1, 10, 100} typically; values are clipped to [0, range_max] then normalized.
        n_bins: Number of equal-width bins (default 10).
        confidences2: Optional second array of confidence values.
        correctness2: Optional second array of correctness values.
        label1: Label for the first dataset in the legend.
        label2: Label for the second dataset in the legend.

    Returns:
        (ax, counts, edges_display): Matplotlib Axes, counts per bin (of first set), and edges in display units.
    """
    # Lazy import to keep module light
    import matplotlib.pyplot as plt

    c = np.asarray(confidences, dtype=float)
    y = np.asarray(correctness, dtype=float)

    rm = float(range_max)
    if rm <= 0:
        rm = 1.0
    c_norm = np.clip(c, 0.0, rm) / rm

    # Same binning logic as compute_ece
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    _, counts, _ = compute_ece(c_norm, c_norm, source_scale=source_scale, n_bins=n_bins)

    # Prepare axes and simple bars
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    lefts = edges[:-1] * rm
    widths = np.diff(edges) * rm

    # Stats for first set
    mean_conf_display = (c_norm.mean() * rm) if c_norm.size else float('nan')
    mean_acc = y.mean() if y.size else float('nan')

    if confidences2 is not None:
        c2 = np.asarray(confidences2, dtype=float)
        c2_norm = np.clip(c2, 0.0, rm) / rm
        _, counts2, _ = compute_ece(c2_norm, c2_norm, source_scale=source_scale, n_bins=n_bins)
        
        mean_conf_display2 = (c2_norm.mean() * rm) if c2_norm.size else float('nan')
        if correctness2 is not None:
            y2 = np.asarray(correctness2, dtype=float)
            mean_acc2 = y2.mean() if y2.size else float('nan')
        else:
            mean_acc2 = float('nan')

        # Plot side-by-side
        w = widths / 2
        bars1 = ax.bar(lefts, counts, width=w, align='edge', edgecolor='black', color="#1f77b4", alpha=0.7, 
                       label=f"{label1}\nMean Conf: {mean_conf_display:.2f}\nMean Acc: {mean_acc:.2f}")
        bars2 = ax.bar(lefts + w, counts2, width=w, align='edge', edgecolor='black', color="#ff7f0e", alpha=0.7,
                       label=f"{label2}\nMean Conf: {mean_conf_display2:.2f}\nMean Acc: {mean_acc2:.2f}")
        
        # Count labels
        for rect, n in zip(bars1, counts):
            if n > 0: ax.text(rect.get_x() + rect.get_width()/2.0, rect.get_height(), f"{int(n)}", ha='center', va='bottom', fontsize=8)
        for rect, n in zip(bars2, counts2):
            if n > 0: ax.text(rect.get_x() + rect.get_width()/2.0, rect.get_height(), f"{int(n)}", ha='center', va='bottom', fontsize=8)
            
        ax.legend()
    else:
        bars = ax.bar(lefts, counts, width=widths, align='edge', edgecolor='black', color="#1f77b4", alpha=0.7)
        # Count labels above each bar
        for rect, n in zip(bars, counts):
            ax.text(rect.get_x() + rect.get_width()/2.0, rect.get_height(), f"n={int(n)}", ha='center', va='bottom')
        
        # Legend: mean confidence and mean accuracy
        ax.legend([f"Mean Confidence: {mean_conf_display:.2f}\nMean Accuracy: {mean_acc:.2f}"])

    # Minimal axes formatting
    ax.set_xlim(0, rm)
    ax.set_xticks(np.linspace(0, rm, n_bins + 1))
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Frequency')
    if title:
        ax.set_title(title)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    ax.grid(True, alpha=0.3)

    return ax, counts, edges * rm



def plot_ece_diagrams(
    evaluation_labels: np.ndarray,
    scores_dict: dict,
    source_scale: int,
    n_bins: int = 10,
    model_name: str = "medgemma-4b-it",
):
    """Plot reliability diagrams for one or more sets of confidence scores.

    Calculates ECE and Brier score for each provided score vector, and plots a
    bar chart of per-bin accuracies against the diagonal of perfect calibration.

    Args:
        evaluation_labels (np.ndarray): Array of 1s (correct) and 0s (incorrect).
        scores_dict (dict): Mapping from method name to numpy array of confidence scores in [0,1].
        n_bins (int): Number of bins to use for the diagrams.
        model_name (str): Label prefix for saved plot files and titles.

    Returns:
        None. Saves PNGs named "{model_name}_{score_name}_ece.png" and displays plots.
    """
    # Import matplotlib lazily to keep this module light-weight
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import Normalize
    if not isinstance(evaluation_labels, np.ndarray):
        print("Error: evaluation_labels must be a numpy array.")
        return
    if not isinstance(scores_dict, dict):
        print("Error: scores_dict must be a dictionary.")
        return

    # Loop through each scoring method provided in the dictionary
    for score_name, confidence_scores in scores_dict.items():
        if len(evaluation_labels) != len(confidence_scores):
            print(f"Skipping '{score_name}': Mismatch in length between evaluation labels ({len(evaluation_labels)}) and scores ({len(confidence_scores)}).")
            continue

        # --- Calculation block for a single plot ---
        final_ece, bin_counts, bin_accuracies = compute_ece(evaluation_labels, confidence_scores, source_scale=source_scale, n_bins=n_bins)
        brier = compute_brier(evaluation_labels, confidence_scores)
        # Compute standard error for bin accuracies (minimal change):
        # SE = sqrt(p*(1-p)/n) for each bin with count n>0, else 0.
        bin_counts_arr = np.array(bin_counts, dtype=float)
        bin_acc_arr = np.array(bin_accuracies, dtype=float)
        with np.errstate(divide='ignore', invalid='ignore'):
            bin_ses = np.sqrt(np.maximum(bin_acc_arr * (1.0 - bin_acc_arr), 0.0) / np.where(bin_counts_arr > 0, bin_counts_arr, np.nan))
        # Replace NaNs (from zero-count bins) with 0 for plotting
        bin_ses = np.nan_to_num(bin_ses, nan=0.0)
        # --- End of Calculation Block ---

        # --- Plotting ---
        print(f"\nPlotting for '{score_name}'...")
        print(f"Calculated ECE: {final_ece:.4f}")
        print(f"Calculated Brier: {brier:.4f}")

        fig, ax = plt.subplots(figsize=(8, 8))
        bar_width = 1.0 / n_bins
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bar_centers = bin_lowers + bar_width / 2

        # Map counts to colors
        counts = np.array(bin_counts, dtype=float)
        # Avoid division by zero if all counts are zero; use 1 as fallback
        norm = Normalize(vmin=0, vmax=counts.max() if counts.max() > 0 else 1.0)
        cmap = cm.Blues  # whiteish blue (low) -> dark blue (high)
        colors = cmap(norm(counts))

        # Draw bars with color indicating density (count)
        bars = ax.bar(
            bar_centers,
            bin_accuracies,
            width=bar_width * 0.9,
            edgecolor='black',
            color=colors,
            yerr=bin_ses,
            capsize=3,
            label='Model Accuracy'
        )

        # Add count labels above each bar
        for i, (rect, count) in enumerate(zip(bars, bin_counts)):
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., height,
                   f'n={count}',
                   ha='center', va='bottom', rotation=0)

        ax.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
        overall_accuracy = np.mean(evaluation_labels)
        ax.set_title(f'{model_name} /w Acc:{overall_accuracy:.3f} & Brier:{brier:.3f}\nReliability Diagram for: {score_name}\nECE = {final_ece:.4f}', fontsize=16)
        ax.set_xlabel('Normalized Confidence', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.legend()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(f'{model_name}_{score_name}_ece.png')
        plt.show()

        return final_ece, brier


def plot_ece_diagrams_similarity(
    evaluation_similarity: np.ndarray,
    scores_dict: dict,
    source_scale: int,
    n_bins: int = 10,
    model_name: str = "medgemma-4b-it",
):
    """Plot reliability diagrams where labels are similarity scores (soft labels).

    This is a copy of `plot_ece_diagrams` that treats `evaluation_similarity` as a
    continuous target in [0, 1] (e.g., text similarity or graded correctness), not a
    binary 0/1 correctness label.

    The plotted per-bin value is the mean similarity in that confidence bin.

    Args:
        evaluation_similarity (np.ndarray): Array of similarity scores in [0,1].
        scores_dict (dict): Mapping from method name to numpy array of confidence scores in [0,1].
        source_scale (int): Scale hint for quantization inside `compute_ece`.
        n_bins (int): Number of bins to use for the diagrams.
        model_name (str): Label prefix for saved plot files and titles.

    Returns:
        Tuple[float, float] | None: (ece, brier) for the first plotted entry; None on invalid input.
    """
    # Import matplotlib lazily to keep this module light-weight
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import Normalize

    if not isinstance(evaluation_similarity, np.ndarray):
        print("Error: evaluation_similarity must be a numpy array.")
        return
    if not isinstance(scores_dict, dict):
        print("Error: scores_dict must be a dictionary.")
        return

    # Loop through each scoring method provided in the dictionary
    for score_name, confidence_scores in scores_dict.items():
        if len(evaluation_similarity) != len(confidence_scores):
            print(
                f"Skipping '{score_name}': Mismatch in length between evaluation_similarity ({len(evaluation_similarity)}) and scores ({len(confidence_scores)})."
            )
            continue

        # --- Calculation block for a single plot ---
        final_ece, bin_counts, bin_means = compute_ece(
            evaluation_similarity,
            confidence_scores,
            source_scale=source_scale,
            n_bins=n_bins,
        )
        brier = compute_brier(evaluation_similarity, confidence_scores)

        # Standard error for the mean similarity in each bin.
        # (For soft labels, the Bernoulli SE formula p*(1-p)/n is not appropriate.)
        sim = np.asarray(evaluation_similarity, dtype=float)
        conf = np.asarray(confidence_scores, dtype=float)
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        bin_ses: list[float] = []
        for i in range(n_bins):
            lo, hi = float(bins[i]), float(bins[i + 1])
            in_bin = (conf >= lo) & (conf < hi) if i < n_bins - 1 else (conf >= lo) & (conf <= hi)
            vals = sim[in_bin]
            n_b = int(vals.size)
            if n_b <= 1:
                bin_ses.append(0.0)
            else:
                bin_ses.append(float(np.std(vals, ddof=1) / np.sqrt(n_b)))
        bin_ses = np.array(bin_ses, dtype=float)
        # --- End of Calculation Block ---

        # --- Plotting ---
        print(f"\nPlotting for '{score_name}'...")
        print(f"Calculated ECE: {final_ece:.4f}")
        print(f"Calculated Brier: {brier:.4f}")

        fig, ax = plt.subplots(figsize=(8, 8))
        bar_width = 1.0 / n_bins
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bar_centers = bin_lowers + bar_width / 2

        # Map counts to colors
        counts = np.array(bin_counts, dtype=float)
        # Avoid division by zero if all counts are zero; use 1 as fallback
        norm = Normalize(vmin=0, vmax=counts.max() if counts.max() > 0 else 1.0)
        cmap = cm.Blues  # whiteish blue (low) -> dark blue (high)
        colors = cmap(norm(counts))

        # Draw bars with color indicating density (count)
        bars = ax.bar(
            bar_centers,
            bin_means,
            width=bar_width * 0.9,
            edgecolor='black',
            color=colors,
            yerr=bin_ses,
            capsize=3,
            label='Mean Similarity'
        )

        # Add count labels above each bar
        for i, (rect, count) in enumerate(zip(bars, bin_counts)):
            height = rect.get_height()
            ax.text(
                rect.get_x() + rect.get_width() / 2.0,
                height,
                f'n={count}',
                ha='center',
                va='bottom',
                rotation=0,
            )

        ax.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
        overall_similarity = float(np.mean(evaluation_similarity))
        ax.set_title(
            f'{model_name} /w MeanSim:{overall_similarity:.3f} & Brier:{brier:.3f}\nReliability Diagram for: {score_name}\nECE = {final_ece:.4f}',
            fontsize=16,
        )
        ax.set_xlabel('Normalized Confidence', fontsize=12)
        ax.set_ylabel('Mean Similarity', fontsize=12)
        ax.legend()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(f'{model_name}_{score_name}_ece_similarity.png')
        plt.show()

        return final_ece, brier


def plot_gain_distribution(
    values: np.ndarray,
    source_scale: int,
    range_max: int = 1,
    ax: Optional['plt.Axes'] = None,
    title: Optional[str] = None,
    y_lim: Optional[Tuple[float, float]] = None,
    n_bins: int = 10,
    ):
    """Plot a distribution of values in [-range_max, range_max] using symmetric binning.

    Uses the same binning logic as plot_confidence_distribution (via compute_ece)
    applied to absolute values, then mirrors for negative values.

    Args:
        values: Array-like values (expected in [-range_max, range_max]).
        source_scale: Scale of source data.
        range_max: Max absolute value for range.
        n_bins: Number of bins per side (positive/negative).

    Returns:
        (ax, counts_pos, counts_neg, edges_display): Matplotlib Axes, counts, and edges.
    """
    # Lazy import to keep module light
    import matplotlib.pyplot as plt

    v = np.asarray(values, dtype=float)
    rm = float(range_max)
    if rm <= 0:
        rm = 1.0
    
    # Normalize to [-1, 1]
    v_norm = np.clip(v, -rm, rm) / rm

    # Split into positive and negative (absolute)
    v_pos = v_norm[v_norm >= 0]
    v_neg = np.abs(v_norm[v_norm < 0])

    # Get counts using compute_ece logic
    # We pass the same array for correctness as it's not used for counts
    _, counts_pos, _ = compute_ece(v_pos, v_pos, source_scale=source_scale, n_bins=n_bins)
    _, counts_neg, _ = compute_ece(v_neg, v_neg, source_scale=source_scale, n_bins=n_bins)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    widths = np.diff(edges) * rm

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    # Plot Positive
    lefts_pos = edges[:-1] * rm
    bars_pos = ax.bar(lefts_pos, counts_pos, width=widths, align='edge', edgecolor='black', color="#1f77b4", alpha=0.7)

    # Plot Negative
    # Negative bins mirror positive ones. 
    # Bin i (0-based) for negatives corresponds to absolute range [edges[i], edges[i+1]].
    # In negative values: [-edges[i+1], -edges[i]].
    # We use align='edge' with positive width, so we need the left edge of the bar.
    # Left edge is -edges[i+1].
    lefts_neg = -edges[1:] * rm
    bars_neg = ax.bar(lefts_neg, counts_neg, width=widths, align='edge', edgecolor='black', color="#d62728", alpha=0.7)

    # Labels
    for rect, n in zip(bars_pos, counts_pos):
        if n > 0:
            ax.text(rect.get_x() + rect.get_width()/2.0, rect.get_height(), f"{int(n)}", ha='center', va='bottom', fontsize=8)
            
    for rect, n in zip(bars_neg, counts_neg):
        if n > 0:
            ax.text(rect.get_x() + rect.get_width()/2.0, rect.get_height(), f"{int(n)}", ha='center', va='bottom', fontsize=8)

    # Legend
    mean_val = v.mean() if v.size else float('nan')
    ax.legend([f"Mean Value: {mean_val:.2f}"])

    ax.set_xlim(-rm, rm)
    # Ticks: symmetric
    ticks = np.concatenate([-edges[1:][::-1], edges]) * rm
    ax.set_xticks(ticks)
    
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    if title:
        ax.set_title(title)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    ax.grid(True, alpha=0.3)
    
    # Add a vertical line at 0
    ax.axvline(0, color='black', linewidth=0.8)

    return ax, counts_pos, counts_neg, edges * rm
