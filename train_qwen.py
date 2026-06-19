"""SFT training script for Qwen2-VL-7B-Instruct with confidence calibration (v12).

Qwen2-VL fork of v9. Hardcoded for Qwen2-VL-7B-Instruct (0-9 confidence scale).

* Dataset: Any dataset with `group_id`, `full_text`, `image_path`, `correct` columns.
* Model: Qwen2-VL-7B-Instruct loaded via `AutoModelForImageTextToText`.
* Loss: Brier loss or EGC (Expected Gradient Calibration) loss.
* Confidence: digit 0-9 immediately after the confidence prefix, divided by 9.
"""

from __future__ import annotations

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import random
from dataclasses import dataclass
from PIL import Image
from typing import Any, Optional, Tuple, List

import pandas as pd
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, PeftModel
from accelerate import Accelerator
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForImageTextToText, AutoProcessor
from tqdm import tqdm

from helpers import (
	load_training_set
)
import numpy as np
import wandb
import transformers
from utils import compute_ece, compute_brier, compute_auroc
print(f"transformers=={transformers.__version__}")


# Constants for answer region detection (KL anchor) — Qwen2-VL tokenizer
# "Rationale:" tokenizes to [49, 37035, 25] ("R", "ationale", ":")
RATIONALE_TOKEN_IDS = [49, 37035, 25]
# End offset: exclude last 5 tokens ("Confidence: X" in Qwen2 tokenizer)
ANSWER_END_OFFSET = 5


@dataclass
class TrainConfig:
	"""Configuration for Qwen2-VL confidence SFT."""

	model_name: str = "Qwen/Qwen2-VL-7B-Instruct"
	output_dir: str = "./outputs_qwen2"

	# Data
	max_length: int = 1024
	train_batch_size: int = 8
	eval_batch_size: int = 8
	num_workers: int = 2
	val_ratio: float = 0.05

	# Training
	num_epochs: int = 1
	lr: float = 1e-4
	weight_decay: float = 0.0
	max_grad_norm: float = 1.0
	seed: int = 42
	use_bf16: bool = True
	use_lora: bool = True
	optimizer: str = "adamw"  # "adamw" or "ademamix"

	# Loss / calibration
	loss_type: str = "brier"  # "brier" or "egc"
	egc_alpha: float = 0.5    # weight for EGC alignment term when loss_type == "egc"
	brier_anchor_lambda: float = 0.5 # Mixing weight for Brier vs Anchor loss (lambda * Brier + (1-lambda) * Anchor)
	wrp_alpha: float = 0.0    # weight for Wrong-Rank Penalty (WRP) loss
	brier_verbalized_only: bool = False  # If True, apply Brier loss only to verbalized samples (index 0)
	softmax_temperature: float = 1.0  # Temperature for softmax on digit logits (higher = flatter distribution)
	align_2x2: bool = True  # If True, use 2x2 causal grid alignment instead of 6-pair alignment (default: True)
	smoke_test: bool = False
	lora_rank: int = 16
	lora_alpha: int = 32
	add_loss_con: bool = False
	target_all: bool = False
	logging_steps: int = 50
	log_gradients: bool = False  # If True, log gradient histograms for loss visualization (wandb)
	gradient_log_steps: int = 200  # Log gradients every N steps (higher = less overhead)
	wandb_project: str = "qwen2_uncertainty_sft"
	wandb_run_name: Optional[str] = None
	use_wandb: bool = True
	sanity_check: bool = False
	no_shuffle: bool = False  # If True, disable training data shuffling (for deterministic visualization)
	dataset: str = "omnimedvqa"
	freeze_vision_encoder: bool = False
	dataset_file: Optional[str] = None  # Direct path to CSV (overrides --dataset)
	ungrouped: bool = False  # If True, use SimpleConfidenceDataset
	resume_adapter: Optional[str] = None  # Path to existing LoRA adapter to resume from

	# KL-anchor regularization (preserves base model answering behavior)
	use_kl_anchor: bool = False
	kl_weight: float = 0.1
	kl_top_k: int = 100
	kl_temperature: float = 3.0  # Temperature for KL softmax (>1 = softer, focuses on ranking over sharpness)


class GroupedConfidenceDataset(Dataset):
	"""Dataset for confidence fine-tuning with grouped perturbation conditions.

	Expects a DataFrame with a `group_id` column where each group has exactly
	4 rows (e.g., baseline + 3 perturbations, or 4 perturbation conditions).
	This is dataset-agnostic and works with any dataset (OmniMedVQA, PMC-VQA, etc.)
	that follows this structure.

	Each __getitem__ returns a list of 4 example dicts; the collate_fn then
	flattens groups so that the model sees a batch of 4 * num_groups examples.
	"""

	def __init__(
		self,
		df: pd.DataFrame,
		processor,
		split: str,
		max_length: int,
		group_col: str = "group_id",
	) -> None:
		self.processor = processor
		self.split = split
		self.max_length = max_length
		self.group_col = group_col

		if group_col not in df.columns:
			raise ValueError(f"GroupedConfidenceDataset expects a '{group_col}' column in the DataFrame.")

		self.groups: List[List[dict[str, Any]]] = []
		for _, gdf in df.groupby(group_col):
			records = gdf.to_dict("records")
			# Only keep well-formed groups of size 4
			if len(records) != 4:
				continue

			group_items: List[dict[str, Any]] = []
			for row in records:
				full_text = row["full_text"]
				try:
					y = float(row["correct"])
				except (ValueError, TypeError):
					continue

				if pd.isna(y):
					continue

				group_items.append({
					"input_text": full_text,
					"image_path": row["image_path"],
					"y": y,
				})

			# Only keep fully valid groups
			if len(group_items) == 4:
				self.groups.append(group_items)

	def __len__(self) -> int:
		return len(self.groups)

	def __getitem__(self, idx: int) -> List[dict[str, Any]]:
		return self.groups[idx]

	def collate_fn(self, batch: List[List[dict[str, Any]]]) -> dict[str, torch.Tensor]:
		"""Collate groups of 4 perturbations into a flat batch.

		If DataLoader(batch_size=N) is used, the effective model batch size
		is 4 * N examples.
		"""

		# Flatten list of groups into a single list of examples
		items: List[dict[str, Any]] = [ex for group in batch for ex in group]

		input_texts = [item["input_text"] for item in items]
		images = [Image.open(item["image_path"]).convert("RGB") for item in items]
		ys = torch.tensor([item["y"] for item in items], dtype=torch.float)

		inputs = tokenize_qwen_batch(
			processor=self.processor,
			input_texts=input_texts,
			images=images,
		)

		batch_out = dict(inputs)
		batch_out["y"] = ys
		return batch_out


class SimpleConfidenceDataset(Dataset):
	"""Ungrouped dataset for plain Brier/CE training.
	
	Unlike GroupedConfidenceDataset, this class does not require groups of 4 samples.
	Each sample is treated independently, suitable for datasets with 1 sample per group.
	"""

	def __init__(
		self,
		df: pd.DataFrame,
		processor,
		split: str,
		max_length: int,
	) -> None:
		self.processor = processor
		self.split = split
		self.max_length = max_length
		self.samples: List[dict[str, Any]] = []

		for _, row in df.iterrows():
			try:
				y = float(row["correct"])
			except (ValueError, TypeError):
				continue
			if pd.isna(y):
				continue
			self.samples.append({
				"input_text": row["full_text"],
				"image_path": row["image_path"],
				"y": y,
			})

	def __len__(self) -> int:
		return len(self.samples)

	def __getitem__(self, idx: int) -> dict[str, Any]:
		return self.samples[idx]

	def collate_fn(self, batch: List[dict[str, Any]]) -> dict[str, torch.Tensor]:
		"""Collate individual samples into a batch."""
		input_texts = [item["input_text"] for item in batch]
		images = [Image.open(item["image_path"]).convert("RGB") for item in batch]
		ys = torch.tensor([item["y"] for item in batch], dtype=torch.float)

		inputs = tokenize_qwen_batch(
			processor=self.processor,
			input_texts=input_texts,
			images=images,
		)

		batch_out = dict(inputs)
		batch_out["y"] = ys
		return batch_out


def tokenize_qwen_batch(
	processor,
	input_texts: List[str],
	images: List[Image.Image],
) -> dict[str, torch.Tensor]:
	"""Tokenize a Qwen2-VL batch without truncating through image-token blocks."""

	common_kwargs = {
		"text": input_texts,
		"return_tensors": "pt",
		"padding": True,
		# Qwen2-VL expands each image placeholder into many visual tokens.
		# Truncating here can cut through that block and trigger
		# "Mismatch in `image` token count between text and `input_ids`".
		"truncation": False,
	}

	try:
		return processor(images=images, **common_kwargs)
	except Exception as flat_exc:
		try:
			nested_images = [[img] for img in images]
			return processor(images=nested_images, **common_kwargs)
		except Exception as nested_exc:
			raise RuntimeError(
				"Qwen2-VL processor failed for both flat and nested image batch formats. "
				f"Flat error: {flat_exc}; Nested error: {nested_exc}"
			)


def split_grouped_dataframe(
	df: pd.DataFrame,
	group_col: str,
	val_ratio: float,
	seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
	"""Split a DataFrame into train/val subsets on whole groups.

	Groups are defined by `group_col` (e.g., `group_id`). This ensures
	that all perturbations of a question stay in the same split and avoids
	partial groups that would later be dropped by GroupedConfidenceDataset.
	
	Splits by group_ids (not sample ids) to keep all samples of each group together.
	"""

	if group_col not in df.columns:
		raise ValueError(f"Grouped split expects column '{group_col}' in DataFrame.")

	if not 0.0 < val_ratio < 1.0:
		raise ValueError("val_ratio must be between 0 and 1.")

	group_ids = df[group_col].unique()
	group_ids = pd.Series(group_ids).sample(frac=1.0, random_state=seed).tolist()
	n_total = len(group_ids)
	n_val = max(1, int(n_total * val_ratio))
	n_train = n_total - n_val
	
	train_groups = set(group_ids[:n_train])
	val_groups = set(group_ids[n_train:])

	df_train = df[df[group_col].isin(train_groups)].reset_index(drop=True)
	df_val = df[df[group_col].isin(val_groups)].reset_index(drop=True)
	return df_train, df_val


def build_dataloaders(
	processor,
	train_config: TrainConfig,
) -> Tuple[DataLoader, Optional[DataLoader]]:
	"""Create train and validation dataloaders.
	
	Supports both grouped (perturbation) and ungrouped (robustness) datasets.
	When --ungrouped is set, uses SimpleConfidenceDataset instead of GroupedConfidenceDataset.
	"""

	# Load dataset: use dataset_file if provided, otherwise fall back to load_training_set
	if train_config.dataset_file is not None:
		print(f"Loading dataset from file: {train_config.dataset_file}")
		df = pd.read_csv(train_config.dataset_file)
	else:
		df = load_training_set(train_config.dataset)
	
	if train_config.smoke_test:
		print("Running smoke test with 256 samples.")
		df = df.head(256)

	if train_config.ungrouped:
		# Ungrouped mode: use SimpleConfidenceDataset with random split
		print(f"Using ungrouped dataset mode (SimpleConfidenceDataset)")
		
		# Simple random train/val split
		n_total = len(df)
		n_val = max(1, int(n_total * train_config.val_ratio))
		df = df.sample(frac=1.0, random_state=train_config.seed).reset_index(drop=True)
		df_train = df.iloc[:-n_val]
		df_val = df.iloc[-n_val:]
		
		train_ds = SimpleConfidenceDataset(
			df_train,
			processor=processor,
			split="train",
			max_length=train_config.max_length,
		)
		eval_ds = SimpleConfidenceDataset(
			df_val,
			processor=processor,
			split="val",
			max_length=train_config.max_length,
		)
		print(f"Train samples: {len(train_ds)}, Val samples: {len(eval_ds)}")
	else:
		# Grouped mode: original behavior
		if "group_id" not in df.columns:
			raise ValueError("Dataset must have a 'group_id' column for grouped perturbation training.")
		
		print(f"Using grouped dataset with 'group_id' column.")
		df_train, df_val = split_grouped_dataframe(df, "group_id", train_config.val_ratio, seed=train_config.seed)
		
		train_ds = GroupedConfidenceDataset(
			df_train,
			processor=processor,
			split="train",
			max_length=train_config.max_length,
		)
		eval_ds = GroupedConfidenceDataset(
			df_val,
			processor=processor,
			split="val",
			max_length=train_config.max_length,
		)
		print(f"Train groups: {len(train_ds)}, Val groups: {len(eval_ds)}")

	train_loader = DataLoader(
		train_ds,
		batch_size=train_config.train_batch_size,
		shuffle=not train_config.no_shuffle,
		num_workers=train_config.num_workers,
		pin_memory=True,
		collate_fn=train_ds.collate_fn,
	)
	eval_loader = DataLoader(
		eval_ds,
		batch_size=train_config.eval_batch_size,
		shuffle=False,
		num_workers=train_config.num_workers,
		pin_memory=True,
		collate_fn=eval_ds.collate_fn,
	)

	return train_loader, eval_loader


def setup_tokenizer_and_model(train_config: TrainConfig):
	"""Load processor and Qwen2-VL model for training."""

	processor = AutoProcessor.from_pretrained(
		train_config.model_name,
		trust_remote_code=True,
		min_pixels=256 * 28 * 28,
		max_pixels=640 * 28 * 28,
	)
	# Ensure padding side is left if needed, though processor usually handles it.
	if hasattr(processor, "tokenizer"):
		processor.tokenizer.padding_side = "left"
		# Truncate from the left to preserve the answer/confidence digit at the end
		processor.tokenizer.truncation_side = "left"
		if processor.tokenizer.pad_token_id is None:
			processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

	model = AutoModelForImageTextToText.from_pretrained(
		train_config.model_name,
		low_cpu_mem_usage=True,
		torch_dtype=torch.bfloat16 if train_config.use_bf16 else torch.float16,
		device_map=None,
		trust_remote_code=True,
	)

	# Resize embeddings if necessary
	tokenizer_len = len(processor.tokenizer) if hasattr(processor, "tokenizer") else len(processor)
	if tokenizer_len > model.get_input_embeddings().weight.shape[0]:
		model.resize_token_embeddings(tokenizer_len)

	if train_config.resume_adapter:
		print(f"Resuming from adapter: {train_config.resume_adapter}")
		# Load the PEFT model from the specified checkpoint
		# is_trainable=True ensures the adapter weights are trainable
		model = PeftModel.from_pretrained(model, train_config.resume_adapter, is_trainable=True)
	elif train_config.use_lora:
		if train_config.target_all:
			target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
		else:
			target_modules = ["q_proj", "v_proj"]

		lora_config = LoraConfig(
			r=train_config.lora_rank,
			lora_alpha=train_config.lora_alpha,
			lora_dropout=0.05,
			bias="none",
			task_type="CAUSAL_LM",
			target_modules=target_modules,
		)
		model = get_peft_model(model, lora_config)
		try:
			model.print_trainable_parameters()
		except AttributeError:
			pass

	if train_config.freeze_vision_encoder:
		print("Freezing vision encoder...")
		for name, param in model.named_parameters():
			if "vision_tower" in name or "vision_model" in name or ".visual." in name:
				param.requires_grad = False

	# Enable gradient checkpointing to save memory (must be after PEFT wrapping)
	# Qwen2-VL requires use_reentrant=False to avoid CUBLAS errors in rotary_emb
	model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
	# Required for LoRA when gradient checkpointing is enabled (must be after PEFT wrapping)
	if hasattr(model, "enable_input_require_grads"):
		model.enable_input_require_grads()
	
	return processor, model


def build_digit_token_indices(tokenizer, device) -> torch.Tensor:
	"""Build token ID list for digits 0..9 for the given tokenizer."""

	# [TRACE] Step 1: Select the tokens corresponding to the digits.
	digits = [str(i) for i in range(10)]
	digit_token_ids: List[int] = []
	for d in digits:
		ids = tokenizer.encode(d, add_special_tokens=False)
		if not ids:
			raise ValueError(f"Digit '{d}' did not tokenize to any IDs")
		digit_token_ids.append(ids[-1])
	return torch.tensor(digit_token_ids, device=device, dtype=torch.long)


def compute_wrp_loss(conf_grouped: torch.Tensor, label_grouped: torch.Tensor, alpha: float = 5.0) -> torch.Tensor:
	"""Compute Wrong-Rank Penalty (WRP) loss.

	WRP only penalizes when the ranking is WRONG, and does NOT reward correct rankings.
	This breaks the symmetry of standard ranking loss that pushes toward bi-modality.

	The penalty is log-scaled: confident wrong rankings are penalized more heavily.

	Args:
		conf_grouped: Confidence predictions [G, 4] where G is number of groups
		label_grouped: Ground truth labels [G, 4]
		alpha: Scaling factor for log penalty (higher = stronger penalty for large violations)

	Returns:
		WRP loss scalar (always a tensor)
	"""
	device = conf_grouped.device
	L_wrp = torch.tensor(0.0, device=device)
	count = 0

	for i in range(4):
		for j in range(i + 1, 4):
			delta_c = conf_grouped[:, i] - conf_grouped[:, j]  # conf difference
			delta_y = label_grouped[:, i] - label_grouped[:, j]  # label difference

			# Filter out ties (where labels are approximately the same)
			valid = (delta_y.abs() > 0.1)
			if valid.sum() == 0:
				continue

			# Wrong rank penalty:
			# - When ranking is correct: delta_c * sign(delta_y) > 0 → violation = 0
			# - When ranking is wrong: delta_c * sign(delta_y) < 0 → violation > 0
			sign_y = torch.sign(delta_y[valid])
			violation = F.relu(-delta_c[valid] * sign_y)  # 0 when correct, >0 when wrong

			# Log-scale penalty for confident wrong rankings
			# Small violations get small penalty, large violations get larger (but sub-linear) penalty
			pair_loss = violation * torch.log1p(alpha * violation)
			L_wrp = L_wrp + pair_loss.mean()
			count += 1

	if count > 0:
		L_wrp = L_wrp / count
	return L_wrp



def find_answer_token_positions(
	input_ids: torch.Tensor,
	attention_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
	"""Find the LOGIT positions that PREDICT answer tokens in each sequence.

	In causal LM, logits[:, pos, :] predicts input_ids[:, pos+1].
	So to anchor predictions of tokens at [start, end) in input_ids,
	we need logits at positions [start-1, end-1).

	IMPORTANT: This function assumes LEFT padding.
	With left padding, ALL sequences end at position (max_len - 1),
	so "Confidence: X" is always at positions (max_len - 4) to (max_len - 1).

	Searches for the "Rationale:" token sequence [49, 37035, 25] and returns
	positions adjusted for causal LM indexing, excluding "Confidence: X".

	Args:
		input_ids: [B, seq_len] tensor of token IDs
		attention_mask: [B, seq_len] attention mask to find actual sequence lengths

	Returns:
		start_positions: [B] tensor of LOGIT start indices (for predicting first answer token)
		end_positions: [B] tensor of LOGIT end indices (exclusive)
	"""
	batch_size, max_len = input_ids.shape
	device = input_ids.device

	start_positions = torch.zeros(batch_size, dtype=torch.long, device=device)
	end_positions = torch.zeros(batch_size, dtype=torch.long, device=device)

	# With LEFT padding, all sequences end at max_len - 1
	# The "Confidence: X" section (last 4 tokens) is at positions (max_len - 4) to (max_len - 1)
	# So answer region ends at position (max_len - ANSWER_END_OFFSET)
	# With causal LM offset (-1), logit end position is (max_len - ANSWER_END_OFFSET - 1)
	answer_end_in_input = max_len - ANSWER_END_OFFSET
	logit_end_pos = answer_end_in_input - 1  # Causal LM offset

	# Convert marker to tensor for comparison
	marker_len = len(RATIONALE_TOKEN_IDS)
	marker_tensor = torch.tensor(RATIONALE_TOKEN_IDS, device=device, dtype=input_ids.dtype)

	for b in range(batch_size):
		# Search for the substring pattern [49, 37035, 25]
		found_pos = -1
		for pos in range(max_len - marker_len + 1):
			if torch.equal(input_ids[b, pos:pos + marker_len], marker_tensor):
				found_pos = pos
				# Don't break - we want the last occurrence

		if found_pos >= 0:
			# Answer tokens start at (found_pos + marker_len) in input_ids
			# To PREDICT these tokens, we need logits at (found_pos + marker_len - 1)
			# But we need at least position 0, so use max()
			answer_start_in_input = found_pos + marker_len
			start_positions[b] = max(0, answer_start_in_input - 1)
		else:
			# If no marker found, start from beginning (fallback)
			start_positions[b] = 0

		# End position is the same for all samples with left padding
		end_positions[b] = max(start_positions[b].item(), logit_end_pos)

	return start_positions, end_positions


def compute_kl_anchor_loss(
	ft_logits: torch.Tensor,
	base_logits: torch.Tensor,
	input_ids: torch.Tensor,
	attention_mask: torch.Tensor,
	start_positions: torch.Tensor,
	end_positions: torch.Tensor,
	top_k: int = 100,
	temperature: float = 3.0,
) -> torch.Tensor:
	"""Compute KL divergence loss anchoring fine-tuned model to base model.

	Optimized vectorized implementation that replaces per-position Python loop
	with batched tensor operations and masking.

	Computes KL(p_base || p_theta) on top-k tokens from base model,
	averaged over answer token positions.

	Temperature scaling (tau > 1) softens both distributions before comparison,
	making KL focus on ranking (relative token orderings) rather than sharpness.

	Args:
		ft_logits: [B, seq_len, vocab] fine-tuned model logits
		base_logits: [B, seq_len, vocab] base model logits
		input_ids: [B, seq_len] input token IDs (unused but kept for consistency)
		attention_mask: [B, seq_len] attention mask
		start_positions: [B] start of answer tokens per sample
		end_positions: [B] end of answer tokens per sample (exclusive)
		top_k: number of top tokens to consider for KL
		temperature: softmax temperature (>1 = softer distributions, focus on ranking)

	Returns:
		kl_loss: scalar KL divergence loss
	"""
	batch_size, seq_len, vocab_size = ft_logits.shape
	device = ft_logits.device

	# Get region bounds - end is fixed (left padding), start varies
	min_start = start_positions.min().item()
	fixed_end = end_positions[0].item()  # Same for all samples with left padding
	region_len = fixed_end - min_start

	if region_len <= 0:
		return torch.tensor(0.0, device=device, requires_grad=True)

	# Slice answer region: [B, region_len, vocab]
	base_region = base_logits[:, min_start:fixed_end, :]
	ft_region = ft_logits[:, min_start:fixed_end, :]
	attn_region = attention_mask[:, min_start:fixed_end]

	# Create position indices for masking: [region_len]
	pos_indices = torch.arange(min_start, fixed_end, device=device)

	# Valid mask: [B, region_len] - True where position is in sample's answer region
	# pos >= start_positions[b] AND pos < end_positions[b] AND attention_mask == 1
	valid_mask = (
		(pos_indices.unsqueeze(0) >= start_positions.unsqueeze(1)) &
		(pos_indices.unsqueeze(0) < end_positions.unsqueeze(1)) &
		(attn_region == 1)
	)

	# Count valid positions for averaging
	total_valid = valid_mask.sum()
	if total_valid == 0:
		return torch.tensor(0.0, device=device, requires_grad=True)

	# Reshape for batched topk: [B * region_len, vocab]
	base_flat = base_region.reshape(-1, vocab_size)
	ft_flat = ft_region.reshape(-1, vocab_size)

	# Batched topk on base model logits
	k = min(top_k, vocab_size)
	top_k_values, top_k_indices = torch.topk(base_flat, k=k, dim=-1)  # [B * region_len, k]

	# Gather corresponding fine-tuned logits
	ft_top_k_logits = torch.gather(ft_flat, dim=-1, index=top_k_indices)  # [B * region_len, k]

	# Compute softmax/log_softmax with temperature
	base_probs = F.softmax(top_k_values / temperature, dim=-1)  # [B * region_len, k]
	ft_log_probs = F.log_softmax(ft_top_k_logits / temperature, dim=-1)  # [B * region_len, k]

	# Compute per-position KL: sum over k dimension
	# KL(base || ft) = sum(base_probs * (log(base_probs) - ft_log_probs))
	per_pos_kl = (base_probs * (base_probs.log() - ft_log_probs)).sum(dim=-1)  # [B * region_len]

	# Reshape back to [B, region_len] and apply mask
	per_pos_kl = per_pos_kl.reshape(batch_size, region_len)
	masked_kl = per_pos_kl * valid_mask.float()

	# Average over valid positions
	kl_loss = masked_kl.sum() / total_valid

	return kl_loss


def train_loop(train_config: TrainConfig) -> None:
	"""Main training loop with Brier loss on confidence digit only."""

	mixed_precision = "bf16" if train_config.use_bf16 else "fp16"
	accelerator = Accelerator(mixed_precision=mixed_precision)

	if accelerator.is_main_process and train_config.use_wandb:
		wandb.init(
			project=train_config.wandb_project,
			name=train_config.wandb_run_name,
			config=vars(train_config),
			settings=wandb.Settings(_disable_stats=True, console="wrap")
		)
		if train_config.wandb_run_name is None and wandb.run is not None:
			train_config.wandb_run_name = wandb.run.name

	# Seed
	torch.manual_seed(train_config.seed)
	random.seed(train_config.seed)

	processor, model = setup_tokenizer_and_model(train_config)

	train_dataloader, eval_dataloader = build_dataloaders(processor, train_config)

	if train_config.optimizer == "ademamix":
		try:
			from pytorch_optimizer import Ademamix
			optimizer = Ademamix(
				model.parameters(), 
				lr=train_config.lr, 
				weight_decay=train_config.weight_decay
			)
			accelerator.print("Using Ademamix optimizer")
		except ImportError:
			accelerator.print("Warning: pytorch-optimizer not installed. Falling back to AdamW.")
			accelerator.print("Install with: pip install pytorch-optimizer")
			optimizer = torch.optim.AdamW(
				model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay
			)
	else:
		optimizer = torch.optim.AdamW(
			model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay
		)
		accelerator.print("Using AdamW optimizer")

	model, optimizer, train_dataloader = accelerator.prepare(
		model, optimizer, train_dataloader
	)
	if eval_dataloader is not None:
		eval_dataloader = accelerator.prepare(eval_dataloader)

	device = accelerator.device
	digit_indices = build_digit_token_indices(processor.tokenizer, device)

	# KL-anchor: verify LoRA is enabled (required for adapter toggle approach)
	if train_config.use_kl_anchor and not train_config.use_lora:
		raise ValueError("KL-anchor requires LoRA (--use_lora) for adapter toggle. "
						 "Without LoRA, there is no adapter to disable.")

	# Determine autocast dtype
	autocast_dtype = torch.float16
	if train_config.use_bf16:
		if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
			autocast_dtype = torch.bfloat16
			accelerator.print("Using bfloat16 for autocast.")
		else:
			accelerator.print("Warning: BF16 requested but not supported or CUDA unavailable. Falling back to float16.")

	# Pre-compute scores tensor outside the loop
	# Map digits 0-9 to 0.0-1.0 (Qwen2 uses 0-9 scale)
	# [TRACE] Step 2: Define numerical values (scores) to match the tokens.
	scores = torch.arange(10, device=device).float() / 9.0
	scores = scores.unsqueeze(0)
	global_step = 0
	for epoch in range(train_config.num_epochs):
		if train_config.sanity_check:
			print("\n[Sanity Check] Running generation on the first batch to verify image understanding...")

			# 1. Grab a single batch from the training loader
			sanity_batch = next(iter(train_dataloader))

			# 2. Move inputs to device (standard boilerplate from your loop)
			# Note: We remove 'y' because generate() doesn't need labels, only inputs
			if "y" in sanity_batch: 
				sanity_batch.pop("y") 
			sanity_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in sanity_batch.items()}

			# 3. Generate text
			# We use max_new_tokens=50 to let the model talk a bit
			model.eval() # Switch to eval mode for generation
			with torch.no_grad():
				# We use the processor to decode, just like in inference
				# Qwen2-VL expects 'pixel_values', 'input_ids', 'image_grid_thw', etc.
				generated_ids = model.generate(
					**sanity_batch,
					max_new_tokens=50,
					do_sample=False # Greedy decoding for deterministic check
				)

			# 4. Decode and Print the results
			input_text_decoded = processor.batch_decode(sanity_batch["input_ids"][0], skip_special_tokens=False)
			generated_text_decoded = processor.batch_decode(generated_ids[0], skip_special_tokens=True)

			print(f"\n--- INPUT (Text Only) ---\n{input_text_decoded}\n")
			print(f"--- MODEL GENERATION ---\n{generated_text_decoded}\n")
			print("------------------------------------------------\n")

		model.train() # Switch back to train mode

		total_loss = 0.0
		num_steps = 0

		train_preds_list = []
		train_labels_list = []

		train_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{train_config.num_epochs}", disable=not accelerator.is_local_main_process)
		for batch in train_pbar:
			global_step += 1
			y = batch.pop("y").to(device)
			# Move remaining batch items (input_ids, attention_mask, pixel_values, etc.) to device
			batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

			# Debug: Check inputs for NaNs
			if "pixel_values" in batch:
				if torch.isnan(batch["pixel_values"]).any():
					accelerator.print("NaN in pixel_values!")
			
			with torch.autocast(device_type="cuda", dtype=autocast_dtype):
				outputs = model(**batch)
				logits = outputs.logits  # [B, seq_len, vocab]
			
			logits = logits.float()

			if torch.isnan(logits).any() or torch.isinf(logits).any():
				accelerator.print(f"Epoch {epoch} Step {num_steps}: NaN or Inf in logits")

			# Optional CE loss over the language modeling tokens (regularizer).
			# In grouped mode: only calculated for verbalized samples (first of each group of 4)
			# In ungrouped mode: applied to all samples
			loss_con = None
			if train_config.add_loss_con:
				if "input_ids" not in batch:
					raise ValueError("Batch is missing 'input_ids' required for CE loss.")
				
				batch_size = logits.size(0)
				
				if train_config.ungrouped:
					# Ungrouped mode: all samples are "verbalized"
					logits_verbalized = logits
					input_ids_verbalized = batch["input_ids"]
					attention_mask_verbalized = batch.get("attention_mask", None)
				else:
					# Grouped mode: extract only verbalized samples (indices 0, 4, 8, 12, ...)
					if batch_size % 4 != 0:
						accelerator.print(f"Warning: Batch size {batch_size} not divisible by 4, skipping CE loss for this batch.")
						logits_verbalized = None
					else:
						verbalized_indices = torch.arange(0, batch_size, 4, device=device)
						logits_verbalized = logits[verbalized_indices]
						input_ids_verbalized = batch["input_ids"][verbalized_indices]
						attention_mask_verbalized = batch["attention_mask"][verbalized_indices] if "attention_mask" in batch else None
				
				if logits_verbalized is not None:
					labels = input_ids_verbalized.clone()
					if attention_mask_verbalized is not None:
						labels[attention_mask_verbalized == 0] = -100

					# Standard causal LM shift: predict token t+1 from token t
					shift_logits = logits_verbalized[:, :-1, :].contiguous()
					shift_labels = labels[:, 1:].contiguous()
					loss_con = F.cross_entropy(
						shift_logits.view(-1, shift_logits.size(-1)),
						shift_labels.view(-1),
						ignore_index=-100,
					)

			# KL-anchor regularization: compute KL(p_base || p_theta) on answer tokens
			loss_kl = None
			if train_config.use_kl_anchor:
				with torch.no_grad():
					with model.disable_adapter():
						with torch.autocast(device_type="cuda", dtype=autocast_dtype):
							base_outputs = model(**batch)
							base_logits = base_outputs.logits.float()

				# Find answer token positions
				start_positions, end_positions = find_answer_token_positions(
					batch["input_ids"],
					batch.get("attention_mask", torch.ones_like(batch["input_ids"])),
				)

					# Compute KL loss on answer tokens
				loss_kl = compute_kl_anchor_loss(
					ft_logits=logits,
					base_logits=base_logits,
					input_ids=batch["input_ids"],
					attention_mask=batch.get("attention_mask", torch.ones_like(batch["input_ids"])),
					start_positions=start_positions,
					end_positions=end_positions,
					top_k=train_config.kl_top_k,
					temperature=train_config.kl_temperature,
				)

			# We are *not* recomputing answers here.
			# y is an offline correctness label from the dataset.
			# We only use logits at the confidence position to compute Brier loss.
			if torch.isnan(logits).any() or torch.isinf(logits).any():
				accelerator.print(f"Logits have NaN or Inf values during evaluation at Epoch {epoch}")
				accelerator.print(f"logits: {logits}")

			# Vectorized extraction of last token logits
			num_token = logits[:, -1, :]
			y_vec = y

			# [TRACE] Step 3: Extract logits for the selected digit tokens.
			num_conf = torch.index_select(num_token, 1, digit_indices)  # [B_eff, 10]
			num_conf = num_conf.float()
			if torch.isnan(num_conf).any() or torch.isinf(num_conf).any():
				accelerator.print(f"Epoch {epoch} Step {num_steps}: NaN or Inf in num_conf (pre-softmax)")
				accelerator.print(f"num_conf (pre-softmax): {num_conf}")
			
			# [TRACE] Step 4: Calculate confidence (probabilities) for the selected tokens.
			# Apply temperature scaling: higher T = flatter distribution, lower T = sharper
			num_conf = torch.softmax(num_conf / train_config.softmax_temperature, dim=1)

			# Retain grad for gradient visualization (only when logging)
			should_log_grads = (
				train_config.log_gradients 
				and global_step % train_config.gradient_log_steps == 0
				and accelerator.is_main_process
			)
			if should_log_grads:
				num_conf.retain_grad()

			# Expand scores to batch size
			scores_expanded = scores.expand(num_conf.size(0), -1)  # [B_eff, 10]

			if train_config.loss_type == "egc":
				# Expected confidence scalar in [0,1] for each example
				conf_scalar = torch.sum(num_conf * scores_expanded, dim=1)  # [B_eff]
				# EGC loss requires batch size divisible by 4 (groups of 4 perturbations)
				if conf_scalar.numel() % 4 != 0 or y_vec.numel() % 4 != 0:
					print(f"Warning: Skipping batch with size {conf_scalar.numel()} (not divisible by 4). Continuing...")
					continue

				conf_grouped = conf_scalar.view(-1, 4)  # [G,4]
				label_grouped = y_vec.view(-1, 4)       # [G,4]

				# Robust term: encourage all 4 conditions' confidences to match the mean correctness
				target = label_grouped.float().mean(dim=1, keepdim=True)  # [G,1]
				L_robust = ((conf_grouped - target) ** 2).mean()

				# Alignment term over all 6 pairs (i < j) of the 4 conditions
				L_align = 0.0
				for i in range(4):
					for j in range(i + 1, 4):
						delta_c = conf_grouped[:, i] - conf_grouped[:, j]
						delta_y = label_grouped[:, i] - label_grouped[:, j]
						L_align = L_align + torch.mean((delta_c - delta_y) ** 2)
				L_align = L_align / 6.0

				loss_cal = L_robust + train_config.egc_alpha * L_align
			elif train_config.loss_type == "log":
				# Log loss (BCE) on robustness with Brier alignment
				# Expected confidence scalar in [0,1] for each example
				conf_scalar = torch.sum(num_conf * scores_expanded, dim=1)  # [B_eff]
				
				# Log loss requires batch size divisible by 4 (groups of 4 perturbations)
				if conf_scalar.numel() % 4 != 0 or y_vec.numel() % 4 != 0:
					print(f"Warning: Skipping batch with size {conf_scalar.numel()} (not divisible by 4). Continuing...")
					continue

				conf_grouped = conf_scalar.view(-1, 4)  # [G,4]
				label_grouped = y_vec.view(-1, 4)       # [G,4]

				# Robust term: BCE loss encouraging confidences to match mean correctness
				target = label_grouped.float().mean(dim=1, keepdim=True)  # [G,1]
				
				# Clamp confidence to avoid log(0)
				eps = 1e-7
				conf_clamped = torch.clamp(conf_grouped, eps, 1.0 - eps)
				
				# BCE: -target * log(c) - (1 - target) * log(1 - c)
				L_robust = -target * torch.log(conf_clamped) - (1.0 - target) * torch.log(1.0 - conf_clamped)
				L_robust = L_robust.mean()

				# Alignment term: keep Brier (MSE) over all 6 pairs
				L_align = 0.0
				for i in range(4):
					for j in range(i + 1, 4):
						delta_c = conf_grouped[:, i] - conf_grouped[:, j]
						delta_y = label_grouped[:, i] - label_grouped[:, j]
						L_align = L_align + torch.mean((delta_c - delta_y) ** 2)
				L_align = L_align / 6.0

				loss_cal = L_robust + train_config.egc_alpha * L_align
			else:
				# Original Brier-style calibration loss over digit distribution
				y_expanded = y_vec.unsqueeze(1).expand_as(scores_expanded)
				squared_diffs = (y_expanded - scores_expanded) ** 2

				# Track alignment metrics (initialized to None for logging)
				L_align_value = None
				L_wrp_value = None
				rank_acc_value = None

				# brier_verbalized_only: apply Brier only to verbalized samples (index 0 in each group of 4)
				if train_config.brier_verbalized_only and not train_config.ungrouped and y_vec.numel() % 4 == 0:
					# Extract only verbalized indices (0, 4, 8, ...)
					verbalized_indices = torch.arange(0, y_vec.numel(), 4, device=y_vec.device)
					brier_per_sample = torch.sum(num_conf * squared_diffs, dim=1)
					loss_cal = brier_per_sample[verbalized_indices].mean()
				else:
					loss_cal = torch.mean(torch.sum(num_conf * squared_diffs, dim=1))

				# Anchor Loss: Expected Squared Error to mid-confidence 0.5
				# New formulation: E[E[(conf - 0.5)²]] = Sum(p_i * (s_i - 0.5)²)
				# This penalizes variance, preventing bimodal distributions that cheat
				# by having E[conf]=0.5 but modes at extremes (e.g., peaks at 0.1 and 0.9)
				L_anchor_value = None
				if train_config.brier_anchor_lambda < 1.0:
					# Squared error for each bin relative to 0.5
					anchor_sq_error = (scores_expanded - 0.5) ** 2  # [B, 10]
					# Weight by predicted probability: E[(conf - 0.5)²]
					L_anchor = torch.sum(num_conf * anchor_sq_error, dim=1).mean()
					loss_cal = train_config.brier_anchor_lambda * loss_cal + (1.0 - train_config.brier_anchor_lambda) * L_anchor
					L_anchor_value = L_anchor.detach().item()
				else:
					# Standard Brier behavior (lambda=1.0)
					pass

				# Optional alignment loss on top of digit distribution Brier
				if (train_config.egc_alpha > 0 or train_config.wrp_alpha > 0) and not train_config.ungrouped:
					if y_vec.numel() % 4 == 0:
						conf_scalar = torch.sum(num_conf * scores_expanded, dim=1)
						conf_grouped = conf_scalar.view(-1, 4)
						label_grouped = y_vec.view(-1, 4)

						# Standard alignment loss (MSE on delta)
						if train_config.egc_alpha > 0:
							if train_config.align_2x2:
								# 2x2 Causal Grid Alignment: 4 main effects instead of 6 pairs
								# Indices: 0=V (verbalized), 1=VC (visual_contrast),
								#          2=V_TS (verbalized + text shuffle), 3=VC_TS (visual_contrast + text shuffle)
								
								# Label effects (in probability space)
								delta_y_img      = label_grouped[:, 0] - label_grouped[:, 1]  # Image effect (original text)
								delta_y_img_ts   = label_grouped[:, 2] - label_grouped[:, 3]  # Image effect (with TS)
								delta_y_ts_img   = label_grouped[:, 2] - label_grouped[:, 0]  # TS effect (with image)
								delta_y_ts_noimg = label_grouped[:, 3] - label_grouped[:, 1]  # TS effect (no image)
								
								# Confidence effects (in probability space)
								delta_c_img      = conf_grouped[:, 0] - conf_grouped[:, 1]
								delta_c_img_ts   = conf_grouped[:, 2] - conf_grouped[:, 3]
								delta_c_ts_img   = conf_grouped[:, 2] - conf_grouped[:, 0]
								delta_c_ts_noimg = conf_grouped[:, 3] - conf_grouped[:, 1]
								
								# 2x2 Alignment Loss: align the 4 causal derivatives
								L_align = (
									((delta_c_img - delta_y_img) ** 2).mean() +
									((delta_c_img_ts - delta_y_img_ts) ** 2).mean() +
									((delta_c_ts_img - delta_y_ts_img) ** 2).mean() +
									((delta_c_ts_noimg - delta_y_ts_noimg) ** 2).mean()
								) / 4.0
								
								# Rank accuracy for 2x2: check if signs match for non-tie effects
								correct_pairs = 0
								total_pairs = 0
								for delta_c, delta_y in [(delta_c_img, delta_y_img), (delta_c_img_ts, delta_y_img_ts),
														  (delta_c_ts_img, delta_y_ts_img), (delta_c_ts_noimg, delta_y_ts_noimg)]:
									non_tie = (delta_y.abs() > 1e-6)
									correct_pairs += ((delta_c * delta_y) > 0)[non_tie].sum()
									total_pairs += non_tie.sum()
							else:
								# Original 6-pair alignment
								L_align = 0.0
								correct_pairs = 0
								total_pairs = 0
								for i in range(4):
									for j in range(i + 1, 4):
										delta_c = conf_grouped[:, i] - conf_grouped[:, j]
										delta_y = label_grouped[:, i] - label_grouped[:, j]
										L_align = L_align + torch.mean((delta_c - delta_y) ** 2)
										# Rank accuracy: correct if signs match, exclude ties
										non_tie = (delta_y.abs() > 1e-6)
										correct_pairs += ((delta_c * delta_y) > 0)[non_tie].sum()
										total_pairs += non_tie.sum()
								L_align = L_align / 6.0

							loss_cal = loss_cal + train_config.egc_alpha * L_align

							# Store for logging
							L_align_value = L_align.detach().item()
							rank_acc_value = (correct_pairs.float() / total_pairs.clamp(min=1)).item() if total_pairs > 0 else None

						# Wrong-Rank Penalty (WRP) - only penalize wrong rankings
						if train_config.wrp_alpha > 0:
							L_wrp = compute_wrp_loss(conf_grouped, label_grouped, alpha=5.0)
							loss_cal = loss_cal + train_config.wrp_alpha * L_wrp
							L_wrp_value = L_wrp.detach().item()

			loss = loss_cal + (loss_con if (loss_con is not None) else 0.0)
			if loss_kl is not None:
				loss = loss + train_config.kl_weight * loss_kl

			# Calculate predicted confidence (for metrics): argmax digit as before
			pred_conf = torch.argmax(num_conf, dim=1).float() / 9.0
			
			# Store local predictions and labels
			train_preds_list.append(pred_conf.detach())
			train_labels_list.append(y_vec.detach())

			if torch.isnan(loss) or torch.isinf(loss):
				accelerator.print(f"Epoch {epoch} Step {num_steps}: NaN or Inf in loss")
				accelerator.print(f"y: {y}")
				accelerator.print(f"num_conf: {num_conf}")

			accelerator.backward(loss)

			# Log gradient visualization for Brier/EGC loss (after backward, before optimizer step)
			if should_log_grads and num_conf.grad is not None and train_config.use_wandb:
				grad = num_conf.grad.detach().cpu().float()
				probs_before = num_conf.detach().cpu().float()
				y_np = y_vec.detach().cpu().numpy()
				
				# Log gradient statistics as scalars (cheap)
				grad_log = {
					"grad/mean": grad.mean().item(),
					"grad/std": grad.std().item(),
					"grad/min": grad.min().item(),
					"grad/max": grad.max().item(),
					"grad/abs_mean": grad.abs().mean().item(),
				}
				
				# Log histograms (wandb handles binning efficiently)
				grad_log["grad/histogram"] = wandb.Histogram(grad.flatten().numpy())
				grad_log["probs/histogram"] = wandb.Histogram(probs_before.flatten().numpy())
				
				# Log a few sample distributions as tables (first 4 samples)
				for i in range(min(4, probs_before.size(0))):
					bins = list(range(10))
					p_i = probs_before[i].numpy()
					g_i = grad[i].numpy()
					y_i = y_np[i]
					
					# Compute approximate "after" distribution (gradient descent direction)
					step_size = 0.1
					p_after_raw = p_i - step_size * g_i
					p_after = np.clip(p_after_raw, 0, 1)
					p_after = p_after / p_after.sum()
					
					table = wandb.Table(
						columns=["bin", "prob_before", "grad", "prob_after", "target"],
						data=[[b, float(p_i[b]), float(g_i[b]), float(p_after[b]), float(y_i)] for b in bins]
					)
					grad_log[f"sample_{i}/dist_table"] = table
				
				wandb.log(grad_log, step=global_step)

			if train_config.max_grad_norm > 0.0:
				accelerator.clip_grad_norm_(model.parameters(), train_config.max_grad_norm)
			optimizer.step()
			optimizer.zero_grad()

			total_loss += loss.detach().float().item()
			num_steps += 1
			train_pbar.set_postfix(loss=loss.detach().float().item())

			if accelerator.is_main_process and train_config.use_wandb and global_step % train_config.logging_steps == 0:
				log_dict = {"train_loss": loss.detach().float().item(), "epoch": epoch, "global_step": global_step}
				# Add alignment metrics if available (brier mode with egc_alpha > 0)
				if train_config.loss_type == "brier" and L_align_value is not None:
					log_dict["L_align"] = L_align_value
				if train_config.loss_type == "brier" and rank_acc_value is not None:
					log_dict["rank_acc"] = rank_acc_value
				if train_config.loss_type == "brier" and L_wrp_value is not None:
					log_dict["L_wrp"] = L_wrp_value
				if train_config.loss_type == "brier" and L_anchor_value is not None:
					log_dict["L_anchor"] = L_anchor_value
				if loss_kl is not None:
					log_dict["train_kl_loss"] = loss_kl.detach().float().item()
				wandb.log(log_dict)

		avg_loss = total_loss / max(1, num_steps)
		accelerator.print(f"Epoch {epoch+1}/{train_config.num_epochs} - train Brier loss: {avg_loss:.4f}")

		if train_preds_list:
			train_preds_local = torch.cat(train_preds_list)
			train_labels_local = torch.cat(train_labels_list)
			
			gathered_preds = accelerator.gather(train_preds_local)
			gathered_labels = accelerator.gather(train_labels_local)

			if accelerator.is_main_process:
				train_preds_np = gathered_preds.detach().cpu().numpy()
				train_labels_np = gathered_labels.detach().cpu().numpy()
				train_ece, bin_counts, bin_accuracies = compute_ece(train_labels_np, train_preds_np, source_scale=9)
				train_brier_metric = compute_brier(train_labels_np, train_preds_np)
				train_auroc = compute_auroc(train_labels_np, train_preds_np)
				accelerator.print(f"Epoch {epoch+1} Train ECE: {train_ece:.4f}, Train Brier (metric): {train_brier_metric:.4f}, Train AUROC: {train_auroc:.4f}")
				accelerator.print(f"Confidence Score Distribution: {bin_counts}")
		
				if train_config.use_wandb:
					wandb.log({
						"train_ece": train_ece,
						"train_brier": train_brier_metric,
						"train_auroc": train_auroc,
						"epoch": epoch + 1
					})

				# Save predictions and labels for plotting
				os.makedirs(train_config.output_dir, exist_ok=True)
				np.save(os.path.join(train_config.output_dir, f"{train_config.wandb_run_name}_train_preds_epoch_{epoch+1}.npy"), train_preds_np)
				np.save(os.path.join(train_config.output_dir, f"{train_config.wandb_run_name}_train_labels_epoch_{epoch+1}.npy"), train_labels_np)

		# Optional evaluation
		if eval_dataloader is not None:
			model.eval()
			eval_loss = 0.0
			eval_steps = 0
			eval_correct_pairs = 0
			eval_total_pairs = 0

			eval_preds_list = []
			eval_labels_list = []

			with torch.no_grad():
				for batch in eval_dataloader:
					y = batch.pop("y").to(device)
					batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

					with torch.autocast(device_type="cuda", dtype=autocast_dtype):
						outputs = model(**batch)
						logits = outputs.logits
					
					logits = logits.float()

					if torch.isnan(logits).any() or torch.isinf(logits).any():
						accelerator.print(f"Eval Epoch {epoch} Step {eval_steps}: NaN or Inf in logits")

					# Vectorized extraction of last token logits
					if torch.isnan(logits).any() or torch.isinf(logits).any():
						accelerator.print(f"Logits have NaN or Inf values during evaluation at Epoch {epoch} Step {eval_steps}")
						accelerator.print(f"logits: {logits}")

					num_token = logits[:, -1, :]
					y_vec = y

					# [TRACE] Step 3 (Eval): Extract logits for the selected digit tokens.
					num_conf = torch.index_select(num_token, 1, digit_indices)
					num_conf = num_conf.float()
					if torch.isnan(num_conf).any() or torch.isinf(num_conf).any():
						accelerator.print(f"Eval Epoch {epoch} Step {eval_steps}: NaN or Inf in num_conf (pre-softmax)")
					
					# [TRACE] Step 4 (Eval): Calculate confidence.
					num_conf = torch.softmax(num_conf, dim=1)

					# Expand scores to batch size
					scores_expanded = scores.expand(num_conf.size(0), -1)

					if train_config.loss_type == "egc":
						conf_scalar = torch.sum(num_conf * scores_expanded, dim=1)  # [B_eff]
						# EGC loss requires batch size divisible by 4 (groups of 4 perturbations)
						if conf_scalar.numel() % 4 != 0 or y_vec.numel() % 4 != 0:
							print(f"Warning: Skipping eval batch with size {conf_scalar.numel()} (not divisible by 4). Continuing...")
							continue

						conf_grouped = conf_scalar.view(-1, 4)
						label_grouped = y_vec.view(-1, 4)

						target = label_grouped.float().mean(dim=1, keepdim=True)
						L_robust = ((conf_grouped - target) ** 2).mean()

						L_align = 0.0
						for i in range(4):
							for j in range(i + 1, 4):
								delta_c = conf_grouped[:, i] - conf_grouped[:, j]
								delta_y = label_grouped[:, i] - label_grouped[:, j]
								L_align = L_align + torch.mean((delta_c - delta_y) ** 2)
						L_align = L_align / 6.0

						loss = L_robust + train_config.egc_alpha * L_align
					elif train_config.loss_type == "log":
						# Log loss (BCE) on robustness with Brier alignment
						conf_scalar = torch.sum(num_conf * scores_expanded, dim=1)  # [B_eff]
						
						if conf_scalar.numel() % 4 != 0 or y_vec.numel() % 4 != 0:
							print(f"Warning: Skipping eval batch with size {conf_scalar.numel()} (not divisible by 4). Continuing...")
							continue

						conf_grouped = conf_scalar.view(-1, 4)
						label_grouped = y_vec.view(-1, 4)

						target = label_grouped.float().mean(dim=1, keepdim=True)
						
						eps = 1e-7
						conf_clamped = torch.clamp(conf_grouped, eps, 1.0 - eps)
						
						L_robust = -target * torch.log(conf_clamped) - (1.0 - target) * torch.log(1.0 - conf_clamped)
						L_robust = L_robust.mean()

						L_align = 0.0
						for i in range(4):
							for j in range(i + 1, 4):
								delta_c = conf_grouped[:, i] - conf_grouped[:, j]
								delta_y = label_grouped[:, i] - label_grouped[:, j]
								L_align = L_align + torch.mean((delta_c - delta_y) ** 2)
						L_align = L_align / 6.0

						loss = L_robust + train_config.egc_alpha * L_align
					else:
						y_expanded = y_vec.unsqueeze(1).expand_as(scores_expanded)
						squared_diffs = (y_expanded - scores_expanded) ** 2
						loss = torch.mean(torch.sum(num_conf * squared_diffs, dim=1))

						# Optional alignment loss on top of digit distribution Brier
						if train_config.egc_alpha > 0 and not train_config.ungrouped:
							if y_vec.numel() % 4 == 0:
								conf_scalar = torch.sum(num_conf * scores_expanded, dim=1)
								conf_grouped = conf_scalar.view(-1, 4)
								label_grouped = y_vec.view(-1, 4)

								L_align = 0.0
								for i in range(4):
									for j in range(i + 1, 4):
										delta_c = conf_grouped[:, i] - conf_grouped[:, j]
										delta_y = label_grouped[:, i] - label_grouped[:, j]
										L_align = L_align + torch.mean((delta_c - delta_y) ** 2)
										# Rank accuracy for eval
										non_tie = (delta_y.abs() > 1e-6)
										eval_correct_pairs += ((delta_c * delta_y) > 0)[non_tie].sum().item()
										eval_total_pairs += non_tie.sum().item()
								L_align = L_align / 6.0

								loss = loss + train_config.egc_alpha * L_align

					# Calculate predicted confidence for metrics: still argmax digit
					pred_conf = torch.argmax(num_conf, dim=1).float() / 9.0

					eval_preds_list.append(pred_conf.detach())
					eval_labels_list.append(y_vec.detach())

					if torch.isnan(loss) or torch.isinf(loss):
						accelerator.print(f"Eval Epoch {epoch} Step {eval_steps}: NaN or Inf in loss")
						accelerator.print(f"y: {y}")
						accelerator.print(f"num_conf: {num_conf}")

					eval_loss += loss.detach().float().item()
					eval_steps += 1

			avg_eval_loss = eval_loss / max(1, eval_steps)
			eval_rank_acc = eval_correct_pairs / max(1, eval_total_pairs) if eval_total_pairs > 0 else None
			accelerator.print(
				f"Epoch {epoch+1}/{train_config.num_epochs} - val Brier loss: {avg_eval_loss:.4f}"
			)

			if eval_preds_list:
				eval_preds_local = torch.cat(eval_preds_list)
				eval_labels_local = torch.cat(eval_labels_list)

				gathered_preds = accelerator.gather(eval_preds_local)
				gathered_labels = accelerator.gather(eval_labels_local)

				if accelerator.is_main_process:
					eval_preds_np = gathered_preds.detach().cpu().numpy()
					eval_labels_np = gathered_labels.detach().cpu().numpy()
					eval_ece, _, _ = compute_ece(eval_labels_np, eval_preds_np, source_scale=9)
					eval_brier_metric = compute_brier(eval_labels_np, eval_preds_np)
					eval_auroc = compute_auroc(eval_labels_np, eval_preds_np)
					rank_acc_str = f", Val Rank Acc: {eval_rank_acc:.4f}" if eval_rank_acc is not None else ""
					accelerator.print(f"Epoch {epoch+1} Val ECE: {eval_ece:.4f}, Val Brier (metric): {eval_brier_metric:.4f}, Val AUROC: {eval_auroc:.4f}{rank_acc_str}")

					if train_config.use_wandb:
						eval_log_dict = {
							"val_ece": eval_ece,
							"val_brier": eval_brier_metric,
							"val_auroc": eval_auroc,
							"val_loss": avg_eval_loss,
							"epoch": epoch + 1
						}
						if eval_rank_acc is not None:
							eval_log_dict["val_rank_acc"] = eval_rank_acc
						wandb.log(eval_log_dict)

					# Save predictions and labels for plotting
					os.makedirs(train_config.output_dir, exist_ok=True)
					np.save(os.path.join(train_config.output_dir, f"{train_config.wandb_run_name}_val_preds_epoch_{epoch+1}.npy"), eval_preds_np)
					np.save(os.path.join(train_config.output_dir, f"{train_config.wandb_run_name}_val_labels_epoch_{epoch+1}.npy"), eval_labels_np)

		# Save checkpoint at the end of each epoch
		accelerator.wait_for_everyone()
		if accelerator.is_main_process:
			run_name = train_config.wandb_run_name if train_config.wandb_run_name else "run"
			# Sanitize run name for file path
			run_name = "".join([c if c.isalnum() or c in "._-" else "_" for c in run_name])
			
			checkpoint_dir = os.path.join(train_config.output_dir, "models", f"{run_name}_epoch_{epoch+1}")
			
			unwrapped_model = accelerator.unwrap_model(model)
			# When using LoRA, save_pretrained automatically saves only the adapter weights
			# (adapter_model.bin) and config, leaving the base model untouched.
			unwrapped_model.save_pretrained(
				checkpoint_dir,
				is_main_process=accelerator.is_main_process,
				save_function=accelerator.save,
			)
			if processor:
				processor.save_pretrained(checkpoint_dir)
			accelerator.print(f"Saved checkpoint to {checkpoint_dir}")

	if accelerator.is_main_process and train_config.use_wandb:
		wandb.finish()


def parse_args() -> TrainConfig:
	"""Parse CLI arguments into TrainConfig."""

	parser = argparse.ArgumentParser(
		description="SFT Qwen2-VL-7B with confidence calibration (Brier or EGC loss)",
	)

	parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-VL-7B-Instruct")
	parser.add_argument("--output_dir", type=str, default="./outputs_qwen2")
	parser.add_argument("--max_length", type=int, default=1024)
	parser.add_argument("--train_batch_size", type=int, default=8)
	parser.add_argument("--eval_batch_size", type=int, default=8)
	parser.add_argument("--num_workers", type=int, default=2)
	parser.add_argument("--val_ratio", type=float, default=0.05)
	parser.add_argument("--num_epochs", type=int, default=1)
	parser.add_argument("--lr", type=float, default=1e-4)
	parser.add_argument("--weight_decay", type=float, default=0.0)
	parser.add_argument("--max_grad_norm", type=float, default=1.0)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--no_bf16", action="store_true", help="Disable bfloat16 training (use float16 instead).")
	parser.add_argument("--no_lora", action="store_false", dest="use_lora", default=True)
	parser.add_argument("--lora_rank", type=int, default=8)
	parser.add_argument("--lora_alpha", type=int, default=32)
	parser.add_argument("--target_all", action="store_true", help="If set, target all linear modules for LoRA. Otherwise only q_proj and v_proj.")
	parser.add_argument("--logging_steps", type=int, default=50, help="Log to wandb every N steps.")
	parser.add_argument("--log_gradients", action="store_true", help="Log gradient histograms for loss visualization.")
	parser.add_argument("--gradient_log_steps", type=int, default=200, help="Log gradients every N steps (higher = less overhead).")
	parser.add_argument("--smoke_test", action="store_true", help="Run a quick smoke test with a small subset of data.")
	parser.add_argument("--add_loss_con", action="store_true", help="Add CE loss on LM tokens as a regularizer in addition to Brier loss.")
	parser.add_argument("--wandb_project", type=str, default="qwen2_uncertainty_sft")
	parser.add_argument("--wandb_run_name", type=str, default=None)
	parser.add_argument("--no_wandb", action="store_false", dest="use_wandb", default=True, help="Disable WandB logging.")
	parser.add_argument("--sanity_check", action="store_true", help="First batch generation verbose")
	parser.add_argument("--no_shuffle", action="store_true", help="Disable training data shuffling (for deterministic visualization).")
	parser.add_argument("--freeze_vision_encoder", action="store_true", help="If set, freeze the vision encoder parameters.")
	parser.add_argument("--dataset", type=str, default="omnimedvqa",
		choices=["omnimedvqa", "pmcvqa"],
		help="Dataset to use for training: omnimedvqa or pmcvqa.")
	parser.add_argument("--loss_type", type=str, default="brier", choices=["brier", "egc", "log"], help="Calibration loss type: 'brier', 'egc', or 'log'.")
	parser.add_argument("--egc_alpha", type=float, default=0.5, help="Weight for the EGC alignment term when loss_type='egc'.")
	parser.add_argument("--wrp_alpha", type=float, default=0.0, help="Weight for Wrong-Rank Penalty (WRP) loss. Only penalizes wrong rankings.")
	parser.add_argument("--brier_verbalized_only", action="store_true", help="Apply Brier loss only to verbalized samples (index 0 in each group).")
	parser.add_argument("--softmax_temperature", type=float, default=1.0, help="Temperature for softmax on digit logits. Higher = flatter distribution.")
	parser.add_argument("--align_2x2", action=argparse.BooleanOptionalAction, default=True,
		help="Use 2x2 causal grid alignment (4 main effects) instead of 6-pair alignment. Default: True. Use --no-align_2x2 to disable.")
	parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "ademamix"], help="Optimizer type: 'adamw' or 'ademamix'.")
	parser.add_argument("--dataset_file", type=str, default=None,
		help="Path to CSV dataset file (overrides --dataset). Use for ungrouped datasets.")
	parser.add_argument("--ungrouped", action="store_true",
		help="Use ungrouped dataset (no group-of-4 requirement). Suitable for robustness datasets.")
	parser.add_argument("--resume_adapter", type=str, default=None,
		help="Path to a PEFT adapter checkpoint to resume training from.")
	parser.add_argument("--brier_anchor_lambda", type=float, default=0.5,
		help="Mixing weight for Brier vs Anchor loss (lambda * Brier + (1-lambda) * Anchor)")
	parser.add_argument("--use_kl_anchor", action="store_true",
		help="Use KL-divergence anchoring to base model on answer tokens.")
	parser.add_argument("--kl_weight", type=float, default=0.1,
		help="Weight for KL-anchor loss term.")
	parser.add_argument("--kl_top_k", type=int, default=100,
		help="Number of top tokens to use for KL computation.")
	parser.add_argument("--kl_temperature", type=float, default=3.0,
		help="Temperature for KL softmax (>1 = softer distributions, focuses on ranking over sharpness).")

	args = parser.parse_args()

	cfg = TrainConfig(
		model_name=args.model_name,
		output_dir=args.output_dir,
		max_length=args.max_length,
		train_batch_size=args.train_batch_size,
		eval_batch_size=args.eval_batch_size,
		num_workers=args.num_workers,
		val_ratio=args.val_ratio,
		num_epochs=args.num_epochs,
		lr=args.lr,
		weight_decay=args.weight_decay,
		max_grad_norm=args.max_grad_norm,
		seed=args.seed,
		use_bf16=not args.no_bf16,
		use_lora=args.use_lora,
		smoke_test=args.smoke_test,
		lora_rank=args.lora_rank,
		lora_alpha=args.lora_alpha,
		add_loss_con=args.add_loss_con,
		target_all=args.target_all,
		logging_steps=args.logging_steps,
		log_gradients=args.log_gradients,
		gradient_log_steps=args.gradient_log_steps,
		wandb_project=args.wandb_project,
		wandb_run_name=args.wandb_run_name,
		use_wandb=args.use_wandb,
	    sanity_check=args.sanity_check,
	    no_shuffle=args.no_shuffle,
	    dataset=args.dataset,
	    freeze_vision_encoder=args.freeze_vision_encoder,
	    loss_type=args.loss_type,
	    egc_alpha=args.egc_alpha,
	    wrp_alpha=args.wrp_alpha,
	    brier_verbalized_only=args.brier_verbalized_only,
	    softmax_temperature=args.softmax_temperature,
	    align_2x2=args.align_2x2,
	    optimizer=args.optimizer,
	    dataset_file=args.dataset_file,
	    ungrouped=args.ungrouped,

	    resume_adapter=args.resume_adapter,
	    brier_anchor_lambda=args.brier_anchor_lambda,
	    use_kl_anchor=args.use_kl_anchor,
	    kl_weight=args.kl_weight,
	    kl_top_k=args.kl_top_k,
	    kl_temperature=args.kl_temperature,
	)
	return cfg


def main() -> None:
	train_config = parse_args()
	train_loop(train_config)


if __name__ == "__main__":
	main()
