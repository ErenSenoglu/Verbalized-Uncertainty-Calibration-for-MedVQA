import os
PROMPTS_MODULE_PATH = os.path.join(os.path.dirname(__file__), "prompts.py")
if not os.path.exists(PROMPTS_MODULE_PATH):
    raise FileNotFoundError(f"Required prompts module not found at: {PROMPTS_MODULE_PATH}")
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from peft import PeftModel

parser = argparse.ArgumentParser(description="Run VQA experiments with configurable methods and quick-test mode")
parser.add_argument("--model_path", type=str, default="google/medgemma-4b-it")
parser.add_argument("--quick_test", action="store_true", help="Limit to a small subset for a fast sanity run")
parser.add_argument("--steer", action="store_true", help="Enable steering: build 5 prompts per sample (neutral + 4 steering)")
parser.add_argument("--topk", action="store_true", help="Use Top-K prompting style instead of vanilla")
parser.add_argument("--lora_path", type=str, default=None, help="Path to LoRA adapter")
parser.add_argument("--dataset", type=str, default="omnimed", choices=["omnimed", "pmcvqa", "medxpertqa", "omnimed_test"], help="Dataset to use: 'omnimed', 'pmcvqa', 'medxpertqa', or 'omnimed_test'")
parser.add_argument("--output_prefix", type=str, default="", help="Prefix to prepend to all output filenames (e.g., 'exp1_lr5e6_' -> 'exp1_lr5e6_verbalized_*.npy')")
parser.add_argument("--greedy", action="store_true", help="Use greedy decoding instead of sampling")
parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature (overrides default; ignored if --greedy is set)")
parser.add_argument("--num_runs", type=int, default=None, help="Override number of runs per sample (default: use method-specific value)")
parser.add_argument("--no_compile", action="store_true", help="Skip torch.compile() to save memory")
parser.add_argument("--no_merge", action="store_true", help="Don't merge LoRA weights (use PEFT directly, saves memory)")
parser.add_argument("--batch_size", type=int, default=48, help="Batch size for inference (default: 48, reduce if OOM)")
parser.add_argument("--filter_from", type=str, default=None,
    help="Path to training dataset CSV - auto-filters test set to matching (image_path, question) pairs")
args = parser.parse_args()

# Define the prompt template
QUICK_TEST = bool(args.quick_test)
STEER_MODE = bool(args.steer)
TOPK_MODE = bool(args.topk)

print(f"Loading model from {args.model_path}...")
tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
try:
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
except Exception:
    processor = None

model = AutoModelForCausalLM.from_pretrained(
    args.model_path,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    trust_remote_code=True
)

if args.lora_path:
    print(f"Loading LoRA adapter from {args.lora_path}...")
    model = PeftModel.from_pretrained(model, args.lora_path)
    if not args.no_merge:
        print("Merging LoRA weights into base model...")
        model = model.merge_and_unload()
        # Clear cache to remove any fragmentation from the merge process
        torch.cuda.empty_cache()
    else:
        print("Skipping merge (using PEFT directly)")

# Inference optimizations
model.eval()
torch.backends.cudnn.benchmark = True
if not args.no_compile:
    print("Compiling model with torch.compile()...")
    model = torch.compile(model, mode="reduce-overhead")
else:
    print("Skipping torch.compile()")

llm = model
TOKENIZER = tokenizer

import math

import os
import zipfile
import json
import itertools
from huggingface_hub import hf_hub_download
from prompts import VANILLA_TEMPLATE, TOPK_TEMPLATE, VANILLA_TEMPLATE_5, TOPK_TEMPLATE_5
import random
import argparse

def load_dataset():
    # --- Configuration ---
    repo_id = "foreverbeliever/OmniMedVQA"
    filename = "OmniMedVQA.zip"
    extract_dir = "./OmniMedVQA_data"
    if os.path.exists(extract_dir):
        return
    """
    Downloads, extracts, and displays samples from a zipped VQA dataset.
    """
    # --- 1. Download the ZIP file from Hugging Face Hub ---
    print(f"Downloading {filename} from {repo_id}...")
    try:
        local_zip_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type='dataset'
        )
        print(f"Download complete. File saved to: {local_zip_path}")
    except Exception as e:
        print(f"An error occurred during download: {e}")
    
    # --- 2. Extract the ZIP file ---
    print(f"\nExtracting files to {extract_dir}...")
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print("Extraction complete.")
    
import pandas as pd
import glob
from utils import create_dataset, create_pmcvqa_dataset
from dataset_providers import create_medxpertqa_dataset, create_omnimed_iid_test_dataset

if args.dataset == "pmcvqa":
    # PMC-VQA: use current working directory (where script is run from) as cache
    cache_dir = os.getcwd()
    dataset = create_pmcvqa_dataset(cache_dir=cache_dir)
    base_path = cache_dir
elif args.dataset == "medxpertqa":
    # MedXpertQA: Hard medical VQA with 5 options
    cache_dir = os.getcwd()
    dataset = create_medxpertqa_dataset(cache_dir=cache_dir)
    base_path = cache_dir
elif args.dataset == "omnimed_test":
    # Load near-IID holdout test set via shared provider
    dataset, base_path = create_omnimed_iid_test_dataset()
else:
    load_dataset()
    # If filter_from is provided, load ALL samples from omnimedvqa.csv
    # (create_dataset applies head(10) filtering which we don't want)
    if args.filter_from:
        import pandas as pd
        omnimedvqa_csv = "./omnimedvqa.csv"
        if not os.path.exists(omnimedvqa_csv):
            raise FileNotFoundError(
                f"ERROR: {omnimedvqa_csv} not found. "
                f"When using --filter_from, omnimedvqa.csv must exist in the current directory. "
                f"This file should contain the full OmniMedVQA dataset (all samples from all JSON files)."
            )
        dataset = pd.read_csv(omnimedvqa_csv)
        print(f"Loaded full OmniMedVQA from {omnimedvqa_csv}: {len(dataset)} samples")
    else:
        dataset = create_dataset()
    base_path = os.path.join("./OmniMedVQA_data", "OmniMedVQA")

    # Auto-filter if training dataset CSV provided
    if args.filter_from:
        import re
        filter_df = pd.read_csv(args.filter_from)

        # Extract questions from verbalized samples only
        verbalized = filter_df[filter_df['method'] == 'verbalized'].copy()

        def extract_question(full_text):
            match = re.search(r'<start_of_image>(.+?)<end_of_turn>', full_text, re.DOTALL)
            return match.group(1).strip() if match else None

        verbalized['_extracted_question'] = verbalized['full_text'].apply(extract_question)

        # Normalize image_path (remove prefix to match dataset format)
        verbalized['_img_norm'] = verbalized['image_path'].str.replace(
            './OmniMedVQA_data/OmniMedVQA/', '', regex=False)

        # Assert uniqueness of composite key
        composite_keys = list(zip(verbalized['_img_norm'], verbalized['_extracted_question']))
        assert len(composite_keys) == len(set(composite_keys)), (
            f"ASSERTION FAILED: (image_path, question) is not unique in {args.filter_from}. "
            f"Found {len(composite_keys)} pairs but only {len(set(composite_keys))} unique. "
            "This filtering logic assumes uniqueness - please investigate the dataset."
        )

        filter_keys = set(composite_keys)
        expected_count = len(filter_keys)

        # Filter dataset using composite key
        before_len = len(dataset)
        mask = dataset.apply(lambda r: (r['image_path'], r['question']) in filter_keys, axis=1)
        dataset = dataset[mask].reset_index(drop=True)

        print(f"Filtered OmniMedVQA: {before_len} -> {len(dataset)} samples (from {args.filter_from})")
        
        # Assert 100% match - all training dataset samples should be found
        assert len(dataset) == expected_count, (
            f"ASSERTION FAILED: Expected {expected_count} samples from training dataset but only matched {len(dataset)}. "
            f"Missing {expected_count - len(dataset)} samples. "
            "Check if image_path or question format differs between training dataset and omnimedvqa.csv"
        )

import math
import numpy as np
def calculate_length_normalized_score(logprobs):
    sum_of_logprobs = sum(lp for lp in logprobs)
    num_tokens = len(logprobs)
    
    # This is the average log probability
    avg_logprob = sum_of_logprobs / num_tokens

    # Convert the average log probability back to a regular probability
    # This final value is the geometric mean of the token probabilities
    score = math.exp(avg_logprob)

    return score

from tqdm import tqdm
from PIL import Image
import time
# CLI args parsed at the top of the file
full_df = dataset.copy()

BATCH_SIZE = args.batch_size
REORDER_SEED = 42       # Seed for deterministic reordering
MAX_REPLACEMENT_TRIES = 15            # Cap for rejection sampling attempts per distractor

STEER_INSTRUCTIONS = ["You should be very cautious and tend to give low confidence on almost all answers.",
                    "Be cautious; avoid giving wrong answers with high confidence.", 
                        "Be confident; avoid giving low confidence on correct answers.",
                        "Be very confident and tend to give high confidence on most answers."]

# Templates are loaded from prompts.py

def build_prompts(
    df: pd.DataFrame,
    base_path: str,
    use_black_image: bool = False,
    shuffle_options: bool = False,
    seed: int | None = None,
    apply_distractor_replacement: bool = False,
    force_letter: str | None = None,
    steer: bool = False,
    task_specific_replacement: bool = True,  # if False, use global pool for distractors
    use_topk: bool = False,
):
    """Build prompts and images for a given method variant.

    Returns:
    - prompts: list of dicts accepted by the generation loop
    - ground_truths: list[str], aligned 1:1 with prompts (duplicated if steer=True)
    - displayed_options: np.ndarray shape [num_prompts, 4] (A,B,C,D texts shown; duplicated if steer=True)
    - questions: np.ndarray shape [num_prompts]
    """
    rng = random.Random(seed)

    # Precompute index pools per question_type for ST4 (distractor replacement)
    # This constrains replacements to come from the same task/question_type where possible.
    all_positions = list(range(len(df)))
    has_qtype = 'question_type' in df.columns
    pool_indices_by_qtype: dict = {}
    if has_qtype:
        # Build a mapping: question_type -> list of row positions (iloc indices)
        for pos, qtype in enumerate(df['question_type']):
            pool_indices_by_qtype.setdefault(qtype, []).append(pos)
    prompts = []
    ground_truths: list[str] = []
    displayed_options: list[list[str]] = []
    questions: list[str] = []

    print(f"[INFO] Building prompts for {len(df)} samples...")
    for idx, (_, row) in enumerate(df.iterrows()):
        if idx % 500 == 0:
            print(f"[INFO] Processing sample {idx}/{len(df)}...")
        question = row['question']
        gt_answer = row['gt_answer']
        image_relative_path = row['image_path']
        full_image_path = os.path.join(base_path, image_relative_path)
        if not os.path.exists(full_image_path):
            continue

        # Prepare options (option text only) — support 4 or 5 options
        has_option_e = 'option_E' in df.columns and str(row.get('option_E', '')).strip() != ''
        final_options = [
            row.get('option_A', ''),
            row.get('option_B', ''),
            row.get('option_C', ''),
            row.get('option_D', ''),
        ]
        if has_option_e:
            final_options.append(row.get('option_E', ''))

        # Replace distractors if requested (keep correct option intact).
        # If task_specific_replacement is True, sample from same question_type pool when available;
        # otherwise sample from the global pool regardless of task.
        if apply_distractor_replacement:
            correct_text = row.get('gt_answer', '')
            try:
                correct_idx = final_options.index(correct_text)
            except ValueError:
                correct_idx = None

            if correct_idx is not None and len(df) > 0:
                # Track used texts within this question to avoid duplicates
                used_texts = {t for t in final_options if isinstance(t, str) and t}
                # Ensure correct is included explicitly
                used_texts.add(correct_text)
                # Determine candidate pool indices based on question_type
                qtype_val = row.get('question_type', None) if has_qtype else None
                if task_specific_replacement and has_qtype and qtype_val in pool_indices_by_qtype:
                    candidate_positions = pool_indices_by_qtype[qtype_val]
                else:
                    candidate_positions = all_positions
                if not candidate_positions:
                    candidate_positions = all_positions
                for i_opt in range(len(final_options)):
                    if i_opt == correct_idx:
                        continue
                    original_distractor = final_options[i_opt]
                    replaced = False
                    for _ in range(MAX_REPLACEMENT_TRIES):
                        # Sample a candidate row from the appropriate pool (same question_type if available)
                        r_idx = rng.choice(candidate_positions)
                        opt_letter = rng.choice(['A', 'B', 'C', 'D'])
                        candidate = df.iloc[r_idx].get(f'option_{opt_letter}', '')
                        if (isinstance(candidate, str) and candidate
                            and candidate != correct_text and candidate not in used_texts):
                            final_options[i_opt] = candidate
                            used_texts.add(candidate)
                            replaced = True
                            break
                    if not replaced:
                        # Keep the original distractor; record it to avoid later duplicates if any
                        if isinstance(original_distractor, str) and original_distractor:
                            used_texts.add(original_distractor)

        # Apply optional shuffling / forcing of displayed options after any replacement
        num_opts = len(final_options)
        letters = 'ABCDE'[:num_opts]
        reassigned: dict[str, str]
        if force_letter is not None:
            # Force the CORRECT option to the requested letter; randomize the rest
            correct_text = gt_answer
            try:
                correct_idx = final_options.index(correct_text)
            except ValueError:
                correct_idx = None

            if correct_idx is not None and force_letter in letters:
                target_pos = letters.index(force_letter)
                remaining_positions = [p for p in range(num_opts) if p != target_pos]
                remaining_old_idx = [i for i in range(num_opts) if i != correct_idx]
                rng.shuffle(remaining_positions)
                rng.shuffle(remaining_old_idx)
                perm = [-1] * num_opts
                perm[target_pos] = correct_idx
                for p, old_i in zip(remaining_positions, remaining_old_idx):
                    perm[p] = old_i
                ordered = [final_options[perm[p]] for p in range(num_opts)]
                reassigned = {letters[i]: ordered[i] for i in range(num_opts)}
            else:
                # Fallback to simple shuffle (or no-shuffle)
                options_list = list(zip(letters, final_options))
                if shuffle_options:
                    rng.shuffle(options_list)
                reassigned = {letters[i]: options_list[i][1] for i in range(num_opts)}
        else:
            options_list = list(zip(letters, final_options))
            if shuffle_options:
                rng.shuffle(options_list)
            reassigned = {letters[i]: options_list[i][1] for i in range(num_opts)}

        try:
            original_image = Image.open(full_image_path).convert("RGB")
        except Exception as e:
            print(f"[WARN] Failed to load image {full_image_path}: {e}")
            continue
        if use_black_image:
            image_size = original_image.size
            image_for_prompt = Image.new('RGB', image_size, (0, 0, 0))
        else:
            image_for_prompt = original_image

        # If steering is enabled, generate 5 prompts per sample with NEUTRAL FIRST, then 4 steering prompts;
        # otherwise, generate a single neutral prompt.
        steer_list = ([""] + [s for s in STEER_INSTRUCTIONS if s]) if steer else [""]
        for steer_instruction in steer_list:
            if num_opts == 5:
                template = TOPK_TEMPLATE_5 if use_topk else VANILLA_TEMPLATE_5
            else:
                template = TOPK_TEMPLATE if use_topk else VANILLA_TEMPLATE

            format_kwargs = dict(
                question=question,
                option_a=reassigned['A'],
                option_b=reassigned['B'],
                option_c=reassigned['C'],
                option_d=reassigned['D'],
                STEER_INSTRUCTION=steer_instruction,
            )
            if num_opts == 5:
                format_kwargs['option_e'] = reassigned['E']

            prompt_text = template.format(**format_kwargs)

            prompt_dict = {
                "prompt": prompt_text,
                "multi_modal_data": {"image": image_for_prompt}
            }
            prompts.append(prompt_dict)
            displayed_options.append([reassigned[l] for l in letters])
            questions.append(question)
            ground_truths.append(gt_answer)

    print(f"[INFO] Built {len(prompts)} prompts successfully.")
    return prompts, ground_truths, np.array(displayed_options, dtype=object), np.array(questions, dtype=object)

# --- 2. RUN INFERENCE IN BATCHES ---
def inference_multiple_runs(all_inputs, method, num_runs=10, temperature=0.3, ground_truths=None):
    # Replicate each input num_runs times
    replicated_inputs = []
    for prompt in all_inputs:
        for _ in range(num_runs):
            replicated_inputs.append(prompt)
    
    all_generations = []
    length_normalized_scores = []
    
    print(f"\nStarting inference with {num_runs} runs per sample, batch size of {BATCH_SIZE}, temperature {temperature}...")
    print(f"Total prompts to process: {len(replicated_inputs)}")
    start_time = time.time()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Warmup: process first batch separately to trigger any lazy compilation
    print("[DEBUG] Running warmup batch to trigger compilation...")
    warmup_start = time.time()
    if len(replicated_inputs) > 0:
        warmup_batch = replicated_inputs[:min(BATCH_SIZE, len(replicated_inputs))]
        warmup_prompts = [b["prompt"] for b in warmup_batch]
        warmup_images = [b["multi_modal_data"]["image"] for b in warmup_batch]
        if processor is not None:
            warmup_images_batch = [[img] for img in warmup_images]
            _ = processor(text=warmup_prompts, images=warmup_images_batch, return_tensors="pt", padding=True)
        print(f"[DEBUG] Warmup complete in {time.time() - warmup_start:.2f}s")
    
    # Helper classes to mimic vLLM output structure
    class MockCompletion:
        def __init__(self, text, token_ids, logprobs):
            self.text = text
            self.token_ids = token_ids
            self.logprobs = logprobs

    class MockOutput:
        def __init__(self, completion):
            self.outputs = [completion]

    # Process all replicated inputs in batches
    batch_times = {"preprocess": [], "generate": [], "decode": []}
    for i in tqdm(range(0, len(replicated_inputs), BATCH_SIZE), desc="Processing Batches"):
        batch = replicated_inputs[i:i + BATCH_SIZE]

        t0 = time.time()
        prompts = [b["prompt"] for b in batch]
        images = [b["multi_modal_data"]["image"] for b in batch]

        # Prepare inputs
        if processor is not None:
            # Wrap images in list of lists as the processor seems to treat a flat list as images for a single example
            images_batch = [[img] for img in images]
            inputs = processor(text=prompts, images=images_batch, return_tensors="pt", padding=True)
        else:
            inputs = tokenizer(prompts, return_tensors="pt", padding=True)

        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        t1 = time.time()
        batch_times["preprocess"].append(t1 - t0)

        # Generate
        # Use inference_mode for slightly better performance/memory than no_grad
        with torch.inference_mode():
            gen_outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=not args.greedy,
                temperature=temperature if not args.greedy else None,
                top_p=1.0 if not args.greedy else None,
                return_dict_in_generate=True,
                output_scores=False,
                use_cache=True,
            )
        t2 = time.time()
        batch_times["generate"].append(t2 - t1)

        # Process outputs
        input_len = inputs["input_ids"].shape[1]
        generated_sequences = gen_outputs.sequences[:, input_len:]

        # Use batch_decode for efficiency (avoids per-sample .tolist() and decode calls)
        decoded_texts = tokenizer.batch_decode(generated_sequences, skip_special_tokens=True)
        for text in decoded_texts:
            length_normalized_scores.append(0.0)
            all_generations.append(text.strip())
        t3 = time.time()
        batch_times["decode"].append(t3 - t2)

        # Print timing every 10 batches
        if len(batch_times["preprocess"]) % 10 == 0:
            print(f"[DEBUG] Avg times - preprocess: {sum(batch_times['preprocess'][-10:])/10:.2f}s, "
                  f"generate: {sum(batch_times['generate'][-10:])/10:.2f}s, "
                  f"decode: {sum(batch_times['decode'][-10:])/10:.2f}s")
    
    # Reshape results back to [num_samples, num_runs]
    num_samples = len(all_inputs)
    
    # Reshape all results
    reshaped_generations = np.array(all_generations).reshape(num_samples, num_runs)
    reshaped_scores = np.array(length_normalized_scores).reshape(num_samples, num_runs)
    
    end_time = time.time()
    print(f"Inference complete in {end_time - start_time:.2f} seconds.")
    
    # Save with run dimension and temperature
    temp_str = str(temperature).replace('.', '_')
    np.save(f'{method}_length_normalized_scores_{num_runs}runs_temp{temp_str}.npy', reshaped_scores)
    np.save(f'{method}_all_generations_{num_runs}runs_temp{temp_str}.npy', reshaped_generations)
    np.save(f'{method}_ground_truths.npy', np.array(ground_truths, dtype=object))
    
    return reshaped_scores, None, None, reshaped_generations, ground_truths

# Run configuration matrix
#
# Overnight 8-config sweep (2 temperatures x 4 method variants):
#   Temps: [0.3, 0.7]
#   Methods (method_name, num_runs, use_black_image, shuffle_options, distractor_replacement, force_letter):
#     1) verbalized,          no shuffle, original image   -> ("verbalized",             5, False, False, False)
#     2) visual_contrast,     no shuffle, black image      -> ("visual_contrast",        5, True,  False, False)
#     3) verbalized_shuffle,  shuffle,     original image  -> ("verbalized_shuffle",     5, False, True,  False)
#     4) visual_contrast_shuf,shuffle,     black image     -> ("visual_contrast_shuffle",5, True,  True,  False)
#
# Notes:
# - Set distractor_replacement=True only if you intend to run Stress Test 4; for standard runs leave it False.
# - method_name is used as the file prefix in saved .npy artifacts; distinct names help keep results organized.
# - To change the temperatures or add more, edit the temperatures list below.
# Use temperature=0 for greedy (file naming only), custom value if provided, otherwise default to 1
if args.greedy:
    temperatures = [0]
elif args.temperature is not None:
    temperatures = [args.temperature]
else:
    temperatures = [1]

methods = [
    # Set the last field (force_letter) to one of 'A','B','C','D' to pin the correct option; None leaves behavior unchanged.
    # Only verbalized case enabled for inference sweep
    # Prefix with args.output_prefix and dataset to customize output filenames
    (f"{args.output_prefix}{args.dataset}_medgemma_verbalized",                         1, False, False, False, None),
]


for temp in temperatures:
    print(f"\n{'='*50}")
    print(f"Running inference with temperature {temp}")
    print(f"{'='*50}")
    for method_name, default_num_runs, use_black, shuffle, distractor_replacement, force_letter in methods:
        # Use --num_runs override if specified, otherwise use method default
        num_runs = args.num_runs if args.num_runs is not None else default_num_runs
        print(f"\nPreparing prompts for method: {method_name} | use_black_image={use_black} | reorder={shuffle}")

        # Select the appropriate dataset for this method without mutating the global full_df
        df_for_method = full_df
        if QUICK_TEST:
            df_for_method = df_for_method[:16]

        prompts, gts, disp_opts, qs = build_prompts(
            df_for_method, base_path,
            use_black_image=use_black,
            shuffle_options=shuffle,
            seed=REORDER_SEED,
            apply_distractor_replacement=distractor_replacement,
            force_letter=force_letter,
            steer=STEER_MODE,
            use_topk=TOPK_MODE,
            task_specific_replacement=True if method_name.endswith("_ts") else False
        )
        # Save displayed options (A,B,C,D texts) to reconstruct letter distributions later
        try:
            np.save(f"{method_name}_displayed_options.npy", disp_opts)
        except Exception as e:
            print(f"[WARN] Could not save displayed options for {method_name}: {e}")
        # Run inference
        length_normalized_scores, all_logprobs, all_decoded_tokens, all_generations, _ = inference_multiple_runs(
            prompts, method_name, num_runs=num_runs, temperature=temp, ground_truths=gts
        )