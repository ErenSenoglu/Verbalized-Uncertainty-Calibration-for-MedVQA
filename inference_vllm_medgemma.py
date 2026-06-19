import os
PROMPTS_MODULE_PATH = os.path.join(os.path.dirname(__file__), "prompts.py")
if not os.path.exists(PROMPTS_MODULE_PATH):
    raise FileNotFoundError(f"Required prompts module not found at: {PROMPTS_MODULE_PATH}")

import argparse
# CLI args
parser = argparse.ArgumentParser(description="Run VQA experiments with configurable methods and quick-test mode")
parser.add_argument("--quick_test", action="store_true", help="Limit to a small subset for a fast sanity run")
parser.add_argument("--steer", action="store_true", help="Enable steering: build 5 prompts per sample (neutral + 4 steering)")
parser.add_argument("--topk", action="store_true", help="Use Top-K prompting style instead of vanilla")
parser.add_argument("--build-prompts-only", action="store_true", help="Only build prompts without running inference")
parser.add_argument("--dataset", type=str, default="omnimed", choices=["omnimed", "omnimed_train", "pmcvqa", "pmcvqa_train", "omnimed_test", "medxpertqa"], help="Dataset to use: 'omnimed', 'pmcvqa', 'omnimed_test', or 'medxpertqa'")
parser.add_argument("--logprobs", action="store_true", help="Enable token-level logprobs collection (slower, default: false)")
parser.add_argument("--chunk-size", type=int, default=100000, help="Chunk size for processing (default: 100000, set lower if OOM)")
parser.add_argument("--method", type=int, default=None, help="Run only specific method index (0-3) for multi-machine parallelization")
parser.add_argument("--output_prefix", type=str, default="", help="Prefix to prepend to all output filenames (e.g., 'steer_' -> 'steer_medxpertqa_medgemma_*.npy')")
parser.add_argument("--num_runs", type=int, default=None, help="Override number of runs per sample (default: use method-specific value)")
parser.add_argument("--max_model_len", type=int, default=2048, help="Maximum model context length (default: 2048)")
parser.add_argument("--greedy", action="store_true", help="Use greedy decoding (temperature=0) instead of sampling")
parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature (overrides default; ignored if --greedy is set)")
parser.add_argument("--model", type=str, default="medgemma", choices=["medgemma", "qwen2"], help="Model to use")
parser.add_argument("--model_path", type=str, default=None, help="Override model path (e.g., merged LoRA model directory)")
args = parser.parse_args()

# Model configurations
MODEL_CONFIGS = {
    "medgemma": {
        "model_id": "google/medgemma-4b-it",
        "default_max_model_len": 2048,
        "extra_llm_kwargs": {},
        "label": "medgemma",
    },
    "qwen2": {
        "model_id": "Qwen/Qwen2-VL-7B-Instruct",
        "default_max_model_len": 4096,
        "extra_llm_kwargs": {"limit_mm_per_prompt": {"image": 1}},
        "label": "qwen2",
        "max_pixels": 501760,  # 640*28*28, matches training resolution
    },
}
MODEL_CFG = MODEL_CONFIGS[args.model]
# Allow overriding model path (e.g., for merged LoRA models)
if args.model_path:
    MODEL_CFG["model_id"] = args.model_path

# Define the prompt template
QUICK_TEST = bool(args.quick_test)
STEER_MODE = bool(args.steer)
TOPK_MODE = bool(args.topk)
BUILD_PROMPTS_ONLY = bool(args.build_prompts_only)

import os
import zipfile
import json
import itertools
if not BUILD_PROMPTS_ONLY:
    from huggingface_hub import hf_hub_download
from prompts import VANILLA_TEMPLATE, TOPK_TEMPLATE, VANILLA_TEMPLATE_5, TOPK_TEMPLATE_5
from prompts import QWEN_VANILLA_TEMPLATE, QWEN_TOPK_TEMPLATE, QWEN_VANILLA_TEMPLATE_5, QWEN_TOPK_TEMPLATE_5
import random


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
    print(f"\
Extracting files to {extract_dir}...")
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print("Extraction complete.")
    
import pandas as pd
import glob
from utils import load_dataset, create_dataset, create_pmcvqa_dataset, create_pmcvqa_train_dataset
from dataset_providers import create_medxpertqa_dataset, create_omnimed_iid_test_dataset

if args.dataset == "omnimed_train":
    # OmniMedVQA training pool: filter to 20k pool, then subsample 4k for training
    hybrid_df = pd.read_csv("hybrid_20k_sample.csv")
    question_id_filter = set(hybrid_df['question_id'].tolist())
    load_dataset()
    dataset = create_dataset(question_id_filter=question_id_filter)
    dataset = dataset.sample(n=4000, random_state=42).reset_index(drop=True)
    print(f"Subsampled to {len(dataset)} training samples")
    base_path = os.path.join("./OmniMedVQA_data", "OmniMedVQA")
elif args.dataset == "pmcvqa":
    # PMC-VQA test set
    cache_dir = os.getcwd()
    dataset = create_pmcvqa_dataset(cache_dir=cache_dir)
    base_path = cache_dir
elif args.dataset == "pmcvqa_train":
    # PMC-VQA train set (for perturbation data collection)
    cache_dir = os.getcwd()
    dataset = create_pmcvqa_train_dataset(cache_dir=cache_dir)
    base_path = cache_dir
elif args.dataset == "omnimed_test":
    # Load near-IID holdout test set via shared provider
    dataset, base_path = create_omnimed_iid_test_dataset()
elif args.dataset == "medxpertqa":
    # MedXpertQA: Hard medical VQA with 5 options
    cache_dir = os.getcwd()
    dataset = create_medxpertqa_dataset(cache_dir=cache_dir)
    base_path = cache_dir
else:
    load_dataset()
    dataset = create_dataset()
    base_path = os.path.join("./OmniMedVQA_data", "OmniMedVQA")

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

full_df = dataset.copy()

BATCH_SIZE = 12
# base_path is set above based on dataset selection
REORDER_SEED = 42       # Seed for deterministic reordering
MAX_REPLACEMENT_TRIES = 15            # Cap for rejection sampling attempts per distractor

STEER_INSTRUCTIONS = ["You should be very cautious and tend to give low confidence on almost all answers.",
                    "Be cautious; avoid giving wrong answers with high confidence.", 
                        "Be confident; avoid giving low confidence on correct answers.",
                        "Be very confident and tend to give high confidence on most answers."]




# Qwen requires a processor for chat template formatting
if args.model == "qwen2":
    from transformers import AutoProcessor
    from qwen_vl_utils import process_vision_info
    _qwen_processor = AutoProcessor.from_pretrained(MODEL_CFG["model_id"], trust_remote_code=True)

if not BUILD_PROMPTS_ONLY:

    from vllm import LLM, SamplingParams

    _max_len = args.max_model_len if args.max_model_len != 2048 else MODEL_CFG["default_max_model_len"]
    llm = LLM(
        model=MODEL_CFG["model_id"],
        max_model_len=_max_len,
        trust_remote_code=True,
        dtype="bfloat16",
        tokenizer_mode="auto",
        gpu_memory_utilization=0.95,
        enforce_eager=MODEL_CFG.get("enforce_eager", False),
        **MODEL_CFG["extra_llm_kwargs"],
    )


def _build_qwen_prompt(text_block, image_for_prompt):
    """Wrap text+image into Qwen chat-template format for vLLM."""
    messages = [
        {"role": "system", "content": "You are a helpful medical imaging assistant."},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_for_prompt, "max_pixels": MODEL_CFG.get("max_pixels", 262144)},
                {"type": "text", "text": text_block},
            ],
        },
    ]
    prompt_text = _qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_data, _ = process_vision_info(messages, return_video_kwargs=False)
    return prompt_text, image_data


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
    - prompts: list of dicts accepted by vLLM.generate
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

    for _, row in df.iterrows():
        question = row['question']
        gt_answer = row['gt_answer']
        image_relative_path = row['image_path']
        full_image_path = os.path.join(base_path, image_relative_path)
        if not os.path.exists(full_image_path):
            print(f"[WARN] Image not found at {full_image_path}, skipping sample.")
            continue

        # Prepare options (option text only)
        option_e = row.get('option_E', '')
        has_five_options = isinstance(option_e, str) and len(option_e) > 0
        
        final_options = [
            row.get('option_A', ''),
            row.get('option_B', ''),
            row.get('option_C', ''),
            row.get('option_D', ''),
        ]
        if has_five_options:
            final_options.append(option_e)

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
                
                num_options = len(final_options)
                for i_opt in range(num_options):
                    if i_opt == correct_idx:
                        continue
                    original_distractor = final_options[i_opt]
                    replaced = False
                    for _ in range(MAX_REPLACEMENT_TRIES):
                        # Sample a candidate row from the appropriate pool (same question_type if available)
                        r_idx = rng.choice(candidate_positions)
                        opt_letter = rng.choice(['A', 'B', 'C', 'D', 'E'] if has_five_options else ['A', 'B', 'C', 'D'])
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

        # Apply optional shuffling / forcing of displayed A-D after any replacement
        letters = 'ABCDE' if has_five_options else 'ABCD'
        reassigned: dict[str, str]
        num_options = len(final_options)
        
        if force_letter is not None:
            # Force the CORRECT option to the requested letter; randomize the rest
            correct_text = gt_answer
            try:
                correct_idx = final_options.index(correct_text)
            except ValueError:
                correct_idx = None

            if correct_idx is not None and force_letter in letters:
                target_pos = letters.index(force_letter)
                remaining_positions = [p for p in range(num_options) if p != target_pos]
                remaining_old_idx = [i for i in range(num_options) if i != correct_idx]
                rng.shuffle(remaining_positions)
                rng.shuffle(remaining_old_idx)
                perm = [-1] * num_options
                perm[target_pos] = correct_idx
                for p, old_i in zip(remaining_positions, remaining_old_idx):
                    perm[p] = old_i
                ordered = [final_options[perm[p]] for p in range(num_options)]
                reassigned = {letters[i]: ordered[i] for i in range(num_options)}
            else:
                # Fallback to simple shuffle (or no-shuffle)
                options_list = list(zip(list(letters), final_options))
                if shuffle_options:
                    rng.shuffle(options_list)
                reassigned = {letters[i]: options_list[i][1] for i in range(num_options)}
        else:
            options_list = list(zip(list(letters), final_options))
            if shuffle_options:
                rng.shuffle(options_list)
            reassigned = {letters[i]: options_list[i][1] for i in range(num_options)}

        if BUILD_PROMPTS_ONLY:
            image_for_prompt = full_image_path
        else:
            original_image = Image.open(full_image_path).convert("RGB")
            if use_black_image:
                image_size = original_image.size
                image_for_prompt = Image.new('RGB', image_size, (0, 0, 0))
            else:
                image_for_prompt = original_image

        # If steering is enabled, generate 5 prompts per sample with NEUTRAL FIRST, then 4 steering prompts;
        # otherwise, generate a single neutral prompt.
        steer_list = ([""] + [s for s in STEER_INSTRUCTIONS if s]) if steer else [""]
        for steer_instruction in steer_list:
            # Select template set based on model (all non-MedGemma models use QWEN plain-text templates)
            if args.model != "medgemma":
                if has_five_options:
                    template = QWEN_TOPK_TEMPLATE_5 if use_topk else QWEN_VANILLA_TEMPLATE_5
                else:
                    template = QWEN_TOPK_TEMPLATE if use_topk else QWEN_VANILLA_TEMPLATE
            else:
                if has_five_options:
                    template = TOPK_TEMPLATE_5 if use_topk else VANILLA_TEMPLATE_5
                else:
                    template = TOPK_TEMPLATE if use_topk else VANILLA_TEMPLATE

            if has_five_options:
                prompt_text = template.format(
                    question=question,
                    option_a=reassigned['A'],
                    option_b=reassigned['B'],
                    option_c=reassigned['C'],
                    option_d=reassigned['D'],
                    option_e=reassigned['E'],
                    STEER_INSTRUCTION=steer_instruction
                )
            else:
                prompt_text = template.format(
                    question=question,
                    option_a=reassigned['A'],
                    option_b=reassigned['B'],
                    option_c=reassigned['C'],
                    option_d=reassigned['D'],
                    STEER_INSTRUCTION=steer_instruction
                )

            # Build prompt dict: model-specific chat-template wrapping
            if args.model == "qwen2":
                if BUILD_PROMPTS_ONLY:
                    # Only apply chat template for text; skip image processing to avoid huge npy files
                    messages = [
                        {"role": "system", "content": "You are a helpful medical imaging assistant."},
                        {"role": "user", "content": [
                            {"type": "image", "image": image_for_prompt, "max_pixels": MODEL_CFG.get("max_pixels", 262144)},
                            {"type": "text", "text": prompt_text},
                        ]},
                    ]
                    qwen_prompt = _qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    prompt_dict = {
                        "prompt": qwen_prompt,
                        "multi_modal_data": {"image": image_for_prompt},
                    }
                else:
                    qwen_prompt, qwen_image = _build_qwen_prompt(prompt_text, image_for_prompt)
                    prompt_dict = {
                        "prompt": qwen_prompt,
                        "multi_modal_data": {"image": qwen_image},
                    }
            else:
                prompt_dict = {
                    "prompt": prompt_text,
                    "multi_modal_data": {"image": image_for_prompt}
                }
            prompts.append(prompt_dict)
            if has_five_options:
                displayed_options.append([reassigned['A'], reassigned['B'], reassigned['C'], reassigned['D'], reassigned['E']])
            else:
                displayed_options.append([reassigned['A'], reassigned['B'], reassigned['C'], reassigned['D']])
            questions.append(question)
            ground_truths.append(gt_answer)

    return prompts, ground_truths, np.array(displayed_options, dtype=object), np.array(questions, dtype=object)

# --- 2. RUN INFERENCE ---
def inference_multiple_runs(all_inputs, method, num_runs=10, temperature=0.3, ground_truths=None, collect_logprobs=False):
    _sp_kwargs = dict(
        temperature=temperature,
        top_p=0.9,
        max_tokens=256,
        logprobs=1 if collect_logprobs else None,
    )
    sampling_params = SamplingParams(**_sp_kwargs)
    
    # Replicate each input num_runs times
    replicated_inputs = []
    for prompt in all_inputs:
        for _ in range(num_runs):
            replicated_inputs.append(prompt)
    
    print(f"\
Starting inference with {num_runs} runs per sample, temperature {temperature}...")
    print(f"Total prompts to process: {len(replicated_inputs)}")
    print(f"Logprobs collection: {'enabled' if collect_logprobs else 'disabled'}")
    start_time = time.time()
    
    # Let vLLM handle all batching internally for better throughput
    outputs = llm.generate(replicated_inputs, sampling_params)
    
    all_generations = []
    length_normalized_scores = []
    all_decoded_tokens = []
    all_logprobs = []
    
    for output in outputs:
        all_generations.append(output.outputs[0].text.strip())
        
        if collect_logprobs and output.outputs[0].logprobs:
            logprobs = []
            decoded_tokens = []
            for item in output.outputs[0].logprobs:
                formatted_out = list(item.values())[0]
                logprobs.append(formatted_out.logprob)
                decoded_tokens.append(formatted_out.decoded_token)
            length_normalized_scores.append(calculate_length_normalized_score(logprobs))
            all_logprobs.append(logprobs)
            all_decoded_tokens.append(decoded_tokens)
        else:
            length_normalized_scores.append(0.0)
            all_logprobs.append([])
            all_decoded_tokens.append([])
    
    # Reshape results back to [num_samples, num_runs]
    num_samples = len(all_inputs)
    
    reshaped_generations = np.array(all_generations).reshape(num_samples, num_runs)
    
    # Only reshape logprobs/tokens if collection was enabled
    if collect_logprobs:
        reshaped_scores = np.array(length_normalized_scores).reshape(num_samples, num_runs)
        reshaped_logprobs = np.array(all_logprobs, dtype=object).reshape(num_samples, num_runs)
        reshaped_tokens = np.array(all_decoded_tokens, dtype=object).reshape(num_samples, num_runs)
    else:
        reshaped_scores = None
        reshaped_logprobs = None
        reshaped_tokens = None
    
    end_time = time.time()
    print(f"Inference complete in {end_time - start_time:.2f} seconds.")
    
    # Save with run dimension and temperature
    temp_str = str(temperature).replace('.', '_')
    if collect_logprobs:
        np.save(f'{method}_length_normalized_scores_{num_runs}runs_temp{temp_str}.npy', reshaped_scores)
        np.save(f'{method}_all_logprobs_{num_runs}runs_temp{temp_str}.npy', reshaped_logprobs)
        np.save(f'{method}_all_decoded_tokens_{num_runs}runs_temp{temp_str}.npy', reshaped_tokens)
    np.save(f'{method}_all_generations_{num_runs}runs_temp{temp_str}.npy', reshaped_generations)
    np.save(f'{method}_ground_truths.npy', np.array(ground_truths, dtype=object))
    
    return reshaped_scores, reshaped_logprobs, reshaped_tokens, reshaped_generations, ground_truths

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
# Use temperature=0 for greedy, custom value if provided, otherwise default to 1
if args.greedy:
    temperatures = [0]
elif args.temperature is not None:
    temperatures = [args.temperature]
else:
    temperatures = [1]
methods = [
    # Set the last field (force_letter) to one of 'A','B','C','D' to pin the correct option; None leaves behavior unchanged.
    (f"{args.output_prefix}{args.dataset}_{MODEL_CFG['label']}_verbalized",                         10, False, False, False, None),
    (f"{args.output_prefix}{args.dataset}_{MODEL_CFG['label']}_visual_contrast",                    10, True,  False, False, None),
    (f"{args.output_prefix}{args.dataset}_{MODEL_CFG['label']}_verbalized_shuffle_replaced_ts",        10, False, True,  True,  None),
    (f"{args.output_prefix}{args.dataset}_{MODEL_CFG['label']}_visual_contrast_shuffle_replaced_ts",   10, True,  True,  True,  None),
]

# Filter methods if --method specified (for multi-machine parallelization)
if args.method is not None:
    if 0 <= args.method < len(methods):
        methods = [methods[args.method]]
        print(f"Running only method {args.method}: {methods[0][0]}")
    else:
        raise ValueError(f"--method must be 0-{len(methods)-1}, got {args.method}")

CHUNK_SIZE = args.chunk_size  # Default 100000 (no chunking for 20k), set lower if OOM

for temp in temperatures:
    print(f"\
{'='*50}")
    print(f"Running inference with temperature {temp}")
    print(f"{'='*50}")
    for method_idx, (method_name, default_num_runs, use_black, shuffle, distractor_replacement, force_letter) in enumerate(methods):
        # Use --num_runs override if specified, otherwise use method default
        num_runs = args.num_runs if args.num_runs is not None else default_num_runs
        print(f"\
Preparing prompts for method: {method_name} | use_black_image={use_black} | reorder={shuffle}")

        # Select the appropriate dataset for this method without mutating the global full_df
        df_for_method = full_df
        if QUICK_TEST:
            df_for_method = df_for_method[:16]

        # Process in chunks to avoid OOM from loading too many images
        total_samples = len(df_for_method)
        num_chunks = (total_samples + CHUNK_SIZE - 1) // CHUNK_SIZE
        print(f"Processing {total_samples} samples in {num_chunks} chunk(s) of up to {CHUNK_SIZE}")
        
        all_chunk_generations = []
        all_chunk_gts = []
        all_chunk_disp_opts = []
        all_chunk_prompts = []
        all_chunk_scores = []
        all_chunk_logprobs = []
        all_chunk_tokens = []
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * CHUNK_SIZE
            end_idx = min((chunk_idx + 1) * CHUNK_SIZE, total_samples)
            df_chunk = df_for_method.iloc[start_idx:end_idx].reset_index(drop=True)
            
            print(f"\
  Chunk {chunk_idx + 1}/{num_chunks}: samples {start_idx}-{end_idx-1}")
            
            prompts, gts, disp_opts, qs = build_prompts(
                df_chunk, base_path,
                use_black_image=use_black,
                shuffle_options=shuffle,
                seed=REORDER_SEED,
                apply_distractor_replacement=distractor_replacement,
                force_letter=force_letter,
                steer=STEER_MODE,
                use_topk=TOPK_MODE,
                task_specific_replacement=True if method_name.endswith("_ts") else False
            )
            
            all_chunk_gts.extend(gts)
            all_chunk_disp_opts.append(disp_opts)
            
            if BUILD_PROMPTS_ONLY:
                # Save complete prompt dicts with image paths (not PIL Image objects)
                # Convert PIL Image back to path string for serialization
                prompts_for_save = []
                for p in prompts:
                    prompt_dict = {
                        "prompt": p["prompt"],
                        "multi_modal_data": {"image": p["multi_modal_data"]["image"]}  # This is already a path string in BUILD_PROMPTS_ONLY mode
                    }
                    prompts_for_save.append(prompt_dict)
                all_chunk_prompts.extend(prompts_for_save)
                print(f"  Built {len(prompts)} prompts. Skipping inference.")
                continue
            
            # Run inference on this chunk
            scores, logprobs, tokens, generations, _ = inference_multiple_runs(
                prompts, method_name, num_runs=num_runs, temperature=temp, 
                ground_truths=gts, collect_logprobs=args.logprobs
            )
            
            all_chunk_generations.append(generations)
            all_chunk_scores.append(scores)
            all_chunk_logprobs.append(logprobs)
            all_chunk_tokens.append(tokens)
        
        # Combine all chunks and save
        if BUILD_PROMPTS_ONLY:
            # Save prompts, ground truths, and displayed options
            final_prompts = np.array(all_chunk_prompts, dtype=object)
            final_disp_opts = np.concatenate(all_chunk_disp_opts, axis=0)
            final_gts = np.array(all_chunk_gts, dtype=object)
            
            np.save(f'{method_name}_prompts.npy', final_prompts)
            np.save(f'{method_name}_ground_truths.npy', final_gts)
            np.save(f'{method_name}_displayed_options.npy', final_disp_opts)
            print(f"Saved prompts for {method_name}: {len(final_prompts)} samples")
            continue
            
        # Concatenate results from all chunks
        final_generations = np.concatenate(all_chunk_generations, axis=0)
        final_disp_opts = np.concatenate(all_chunk_disp_opts, axis=0)
        final_gts = np.array(all_chunk_gts, dtype=object)
        
        # Save combined results
        temp_str = str(temp).replace('.', '_')
        np.save(f'{method_name}_all_generations_{num_runs}runs_temp{temp_str}.npy', final_generations)
        np.save(f'{method_name}_ground_truths.npy', final_gts)
        np.save(f'{method_name}_displayed_options.npy', final_disp_opts)
        
        if args.logprobs:
            final_scores = np.concatenate(all_chunk_scores, axis=0)
            final_logprobs = np.concatenate(all_chunk_logprobs, axis=0)
            final_tokens = np.concatenate(all_chunk_tokens, axis=0)
