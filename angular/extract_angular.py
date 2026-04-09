"""
Angular Activation Steering - Direction Extraction

Prompt-conditioned extraction of refusal directions and steering-plane
construction following the Angular activation steering method.

Algorithm summary
-----------------
1. Partition prompts by `should_abstain` label -> two populations.
2. Tokenize prompts (chat template, left-padded) and run through the model
   without generating any response tokens.
3. At each decoder layer, capture activations at two residual-stream sites:
   - `resid_mid`:  residual stream after self-attention + skip, before MLP
                     (operationally: input to the post-attention LayerNorm)
   - `resid_post`: residual stream after the full block (attention + MLP + skip)
4. Extract activations at the template suffix token(s) - the shared tail
   tokens that follow the variable user content (e.g. `<|im_start|>assistant\\n`).
5. Per-token L2-normalize, then pool over suffix positions (last or mean) ->
   one vector per (sample, layer, site).
6. Candidate direction = mean_normed(abstain) - mean_normed(answer) per
   (layer, site).  L2-normalize each candidate.
7. Select best direction u1 (highest mean cosine with other candidates).
8. Build orthogonal second basis u2 via PCA in the orthogonal complement
   of u1.
9. Save `{u1, u2, candidates, metadata}` as a `.pt` file.

Usage
-----
Local (Mac / CPU / MPS):

    python angular/extract_angular.py \\
        --model_name Qwen/Qwen2.5-0.5B-Instruct \\
        --data_path data/abstention_training_dataset.json \\
        --output_path data/angular_vectors/Qwen2_5_0_5B/angular_steering.pt \\
        --use_system_prompt \\
        --max_samples 32 \\
        --batch_size 4

HPC (CUDA):

    python angular/extract_angular.py \\
        --model_name Qwen/Qwen2.5-7B-Instruct \\
        --data_path data/abstention_training_dataset.json \\
        --output_path data/angular_vectors/Qwen2_5_7B/angular_steering.pt \\
        --use_system_prompt \\
        --max_samples 512 \\
        --batch_size 16
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA
from torch.nn.functional import normalize
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from recipe.system_prompt import SYSTEM_PROMPT
import ast

logger = logging.getLogger(__name__)

# candidates with a raw (pre-normalisation) L2 norm below this threshold are
# considered noise and excluded from u1 selection and PCA.
DEFAULT_NORM_FLOOR = 0.01


# data loading


def load_abstention_dataset(
    path: str,
    max_samples: Optional[int] = None,
    exclude_tasks: Optional[set] = None,
    seed: int = 42,
    dedupe: bool = True,
    stratified: bool = False,
) -> Tuple[List[str], List[str]]:
    """Load and partition prompts by `should_abstain`.

    Parameters
    ----------
    dedupe : bool
        Deduplicate by (question, should_abstain) before subsampling.
        The dataset contains multiple (positive, negative) response pairs per
        question, but extraction only uses the question text.  Without dedup,
        identical prompts are counted multiple times, wasting compute and
        corrupting subsampled distributions.

    stratified : bool
        If True and max_samples is set, subsample proportionally by task type
        instead of uniformly at random.  Useful when the task distribution is
        very skewed (e.g. `underspecified context` dominates).

    Returns
    -------
    abstain_prompts, answer_prompts : list[str], list[str]
        Two disjoint lists of question strings.
    """
    with open(path) as f:
        data = json.load(f)

    if exclude_tasks:
        before = len(data)
        data = [d for d in data if d["task"] not in exclude_tasks]
        logger.info(
            "Excluded tasks %s: %d -> %d entries", exclude_tasks, before, len(data)
        )

    # deduplicate by (question, should_abstain) - each unique prompt appears
    # multiple times in the dataset due to paired positive/negative responses
    if dedupe:
        seen: set = set()
        deduped: list = []
        for d in data:
            key = (d["question"].strip(), bool(d["should_abstain"]))
            if key not in seen:
                seen.add(key)
                deduped.append(d)

        logger.info("Deduped prompt-label pairs: %d -> %d", len(data), len(deduped))
        data = deduped

    abstain_data = [d for d in data if d["should_abstain"]]
    answer_data = [d for d in data if not d["should_abstain"]]

    logger.info(
        "Dataset: %d should-abstain, %d should-answer prompts",
        len(abstain_data),
        len(answer_data),
    )

    # log per-task counts by class
    for label, subset in [("abstain", abstain_data), ("answer", answer_data)]:
        task_counts = Counter(d["task"] for d in subset)
        logger.debug("%s task distribution:", label)
        for task, count in sorted(task_counts.items(), key=lambda x: -x[1]):
            logger.debug("%-30s %5d", task, count)

    # sub-sample if requested (deterministic)
    if max_samples is not None:
        rng = np.random.default_rng(seed)
        abstain_data = _subsample(
            abstain_data, max_samples, rng, stratified=stratified, label="abstain"
        )

        answer_data = _subsample(
            answer_data, max_samples, rng, stratified=stratified, label="answer"
        )

    abstain_prompts = [d["question"] for d in abstain_data]
    answer_prompts = [d["question"] for d in answer_data]

    # check for prompt overlap between classes
    overlap = set(abstain_prompts) & set(answer_prompts)
    if overlap:
        logger.warning(
            "%d prompts appear in BOTH classes! "
            "This may indicate label noise or dataset collisions. "
            "Examples: %s",
            len(overlap),
            list(overlap)[:3],
        )
    else:
        logger.info("Prompt overlap check: no overlap between classes (good)")

    return abstain_prompts, answer_prompts


def _subsample(
    data: List[dict],
    max_samples: int,
    rng: np.random.Generator,
    stratified: bool = False,
    label: str = "",
) -> List[dict]:
    """Subsample a list of dataset entries, optionally stratified by task."""
    if len(data) <= max_samples:
        return data

    if not stratified:
        idx = rng.choice(len(data), max_samples, replace=False)
        result = [data[i] for i in sorted(idx)]
    else:
        # stratified: sample proportionally by task
        by_task: Dict[str, list] = {}
        for d in data:
            by_task.setdefault(d["task"], []).append(d)

        result = []
        total = len(data)
        for task, entries in sorted(by_task.items()):
            n_task = max(1, round(max_samples * len(entries) / total))
            n_task = min(n_task, len(entries))
            idx = rng.choice(len(entries), n_task, replace=False)
            result.extend(entries[i] for i in sorted(idx))

        # trim if rounding produced too many
        if len(result) > max_samples:
            idx = rng.choice(len(result), max_samples, replace=False)
            result = [result[i] for i in sorted(idx)]

    logger.info(
        "Sub-sampled %s: %d -> %d prompts%s",
        label,
        len(data),
        len(result),
        " (stratified)" if stratified else "",
    )
    return result


# template suffix detection


def get_template_suffix_tokens(
    tokenizer: AutoTokenizer,
    use_system_prompt: bool,
) -> Tuple[List[str], int]:
    """Detect the shared tail tokens after the variable user content.

    Tokenizes two dummy single-character prompts and scans backwards to find
    where they diverge.  Everything after the divergence is the template suffix
    (e.g.  `<|im_start|>assistant\\n`).

    Returns
    -------
    suffix_token_strs : list[str]
        Human-readable token strings (for logging).

    num_suffix_tokens : int
        Number of suffix tokens.  At least 1 (we always use the last token).
    """

    def _tokenize(text: str) -> torch.Tensor:
        if use_system_prompt:
            msgs = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ]
        else:
            msgs = [{"role": "user", "content": text}]
        # Use tokenize=True for a single-pass tokenization that preserves
        # exact token boundaries (consistent with the CAA script).
        ids = tokenizer.apply_chat_template(
            msgs,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        # apply_chat_template may return BatchEncoding or raw tensor
        if hasattr(ids, "input_ids"):
            ids = ids.input_ids
        return ids.squeeze(0)

    toks_a = _tokenize("a")
    toks_b = _tokenize("b")

    # scan from the end to find the first position where tokens differ.
    suffix_start = len(toks_a)
    min_len = min(len(toks_a), len(toks_b))
    for i in range(1, min_len + 1):
        if toks_a[-i] != toks_b[-i]:
            suffix_start = len(toks_a) - i + 1
            break

    suffix_ids = toks_a[suffix_start:]
    suffix_strs = tokenizer.convert_ids_to_tokens(suffix_ids.tolist())

    # ensure at least 1 token (the very last position)
    if len(suffix_strs) == 0:
        suffix_strs = ["<last_token>"]
        num_suffix = 1
    else:
        num_suffix = len(suffix_strs)

    logger.info("Template suffix tokens (%d): %s", num_suffix, suffix_strs)
    return suffix_strs, num_suffix


# prompt tokenization


def prompts_to_chat_batch(
    tokenizer: AutoTokenizer,
    prompts: List[str],
    use_system_prompt: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Tokenize a list of prompts with the chat template, left-padded.

    Returns
    -------
    input_ids : Tensor of shape (batch, seq_len)

    attention_mask : Tensor of shape (batch, seq_len)
        0 for padding tokens, 1 for real tokens.
    """
    conversations = []
    for p in prompts:
        if use_system_prompt:
            conversations.append(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": p},
                ]
            )
        else:
            conversations.append([{"role": "user", "content": p}])

    result = tokenizer.apply_chat_template(
        conversations,
        padding=True,
        truncation=False,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    # transformers >=4.x returns BatchEncoding with input_ids + attention_mask
    if hasattr(result, "input_ids"):
        input_ids = result.input_ids
        attention_mask = result.attention_mask
    else:
        # fallback: raw tensor (older transformers) - construct mask manually
        input_ids = result
        attention_mask = (input_ids != tokenizer.pad_token_id).long()

    return input_ids, attention_mask


# activation extraction


def _detect_mid_layernorm_pattern(model) -> str:
    """Auto-detect the LayerNorm module name used for resid_mid.

    Returns a format string with `{layer_idx}` placeholder.
    """
    # inspect the first decoder layer's named children
    first_layer = None
    for name, module in model.named_modules():
        # common patterns: model.layers.0, model.model.layers.0
        if re.search(r"layers[.\[]0[.\]]?$", name) or name.endswith("layers.0"):
            first_layer = (name, module)
            break

    if first_layer is None:
        raise RuntimeError(
            "Cannot find decoder layers. Supported architectures: "
            "Llama, Qwen, Gemma, Mistral"
        )

    base_path, layer_module = first_layer

    child_names = [n for n, _ in layer_module.named_children()]

    # gemma uses pre_feedforward_layernorm
    if "pre_feedforward_layernorm" in child_names:
        pattern = base_path.replace(".0", ".{layer_idx}") + ".pre_feedforward_layernorm"
    # llama / qwen / mistral use post_attention_layernorm
    elif "post_attention_layernorm" in child_names:
        pattern = base_path.replace(".0", ".{layer_idx}") + ".post_attention_layernorm"
    else:
        raise RuntimeError(
            f"Cannot identify resid_mid layernorm. "
            f"Children of first layer: {child_names}"
        )

    logger.info("Detected resid_mid layernorm pattern: %s", pattern)
    return pattern


def _detect_decoder_layer_pattern(model) -> str:
    """Auto-detect decoder block path for capturing resid_post via hooks.

    Returns a format string with `{layer_idx}` placeholder.
    """
    for name, _ in model.named_modules():
        # common patterns: model.layers.0, model.model.layers.0
        if re.search(r"layers[.\[]0[.\]]?$", name) or name.endswith("layers.0"):
            pattern = name.replace(".0", ".{layer_idx}")
            logger.info("Detected decoder layer pattern: %s", pattern)
            return pattern

    raise RuntimeError(
        "Cannot find decoder layers. Supported architectures: "
        "Llama, Qwen, Gemma, Mistral"
    )


def _pool_suffix(
    acts: torch.Tensor,
    num_suffix_tokens: int,
    suffix_pool: str,
    do_normalize: bool = True,
) -> torch.Tensor:
    """Extract and pool suffix-token activations.

    Parameters
    ----------
    acts : Tensor of shape (batch, seq, hidden)

    num_suffix_tokens : int

    suffix_pool : str
        `last` - use only the very last token position.
        `mean` - average over the last `num_suffix_tokens` positions.

    do_normalize : bool
        L2-normalize each token before pooling.

    Returns
    -------
    Tensor of shape (batch, hidden)
    """
    if suffix_pool == "last":
        suffix = acts[:, -1:, :].float()
    else:
        suffix = acts[:, -num_suffix_tokens:, :].float()

    if do_normalize:
        suffix = normalize(suffix, dim=-1)

    # mean over suffix tokens -> (batch, hidden)
    return suffix.mean(dim=1)


def extract_activations(
    model,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    use_system_prompt: bool,
    num_suffix_tokens: int,
    batch_size: int,
    device: torch.device,
    suffix_pool: str = "last",
    normalize_acts: bool = True,
) -> Dict[Tuple[int, str], torch.Tensor]:
    """Extract mean activations at (layer, site) for a set of prompts.

    Processes prompts in batches.  For each batch:
    - Registers pre-forward hooks on LayerNorm modules to capture `resid_mid`.
    - Registers forward hooks on decoder blocks to capture `resid_post`.
    - Extracts the suffix position(s) from each.
    - L2-normalizes per token (if `normalize_acts`), then pools.

    Returns a dict mapping `(layer_idx, "resid_mid"|"resid_post")` to a
    tensor of shape `(hidden_dim,)` - the population mean.

    Memory-efficient: accumulates running sums instead of storing per-sample
    activations.
    """
    num_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    ln_pattern = _detect_mid_layernorm_pattern(model)
    decoder_pattern = _detect_decoder_layer_pattern(model)

    # running accumulators: (layer, site) -> running sum of (normalised) vectors
    running_sum: Dict[Tuple[int, str], torch.Tensor] = {}
    for layer_idx in range(num_layers):
        for site in ("resid_mid", "resid_post"):
            running_sum[(layer_idx, site)] = torch.zeros(
                hidden_dim, dtype=torch.float32
            )
    total_samples = 0
    prompt_lengths: List[int] = []

    for batch_start in tqdm(
        range(0, len(prompts), batch_size),
        desc="Extracting activations",
    ):
        batch_prompts = prompts[batch_start : batch_start + batch_size]

        input_ids, attention_mask = prompts_to_chat_batch(
            tokenizer, batch_prompts, use_system_prompt
        )
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        # track prompt token lengths (excluding padding)
        prompt_lengths.extend(attention_mask.sum(dim=1).tolist())

        # register hooks for resid_mid
        resid_mid_cache: Dict[int, torch.Tensor] = {}
        resid_post_cache: Dict[int, torch.Tensor] = {}
        hooks = []

        for layer_idx in range(num_layers):
            module_name = ln_pattern.format(layer_idx=layer_idx)
            target_module = model.get_submodule(module_name)

            # pre-forward hook: input to the LayerNorm is the resid-mid stream
            def _make_hook(li: int):
                def _hook(module, args):
                    # args[0] is the hidden-states tensor before layernorm
                    # keep in original dtype to save memory
                    x = args[0] if isinstance(args, tuple) else args
                    resid_mid_cache[li] = x.detach()

                return _hook

            hooks.append(target_module.register_forward_pre_hook(_make_hook(layer_idx)))

        # register hooks for resid_post on decoder-layer outputs
        for layer_idx in range(num_layers):
            module_name = decoder_pattern.format(layer_idx=layer_idx)
            target_module = model.get_submodule(module_name)

            def _make_post_hook(li: int):
                def _hook(module, args, output):
                    # decoder block output is resid-post for this layer
                    x = output[0] if isinstance(output, tuple) else output
                    resid_post_cache[li] = x.detach()

                return _hook

            hooks.append(target_module.register_forward_hook(_make_post_hook(layer_idx)))

        # forward pass
        with torch.inference_mode():
            model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        # remove hooks
        for h in hooks:
            h.remove()

        actual_batch = input_ids.shape[0]

        # process resid_mid
        for layer_idx in range(num_layers):
            raw = resid_mid_cache[layer_idx]  # (batch, seq, hidden)
            pooled = _pool_suffix(raw, num_suffix_tokens, suffix_pool, normalize_acts)
            running_sum[(layer_idx, "resid_mid")] += pooled.sum(dim=0).cpu()

        # process resid_post captured directly from decoder-layer outputs
        for layer_idx in range(num_layers):
            hs = resid_post_cache[layer_idx]  # (batch, seq, hidden)
            pooled = _pool_suffix(hs, num_suffix_tokens, suffix_pool, normalize_acts)
            running_sum[(layer_idx, "resid_post")] += pooled.sum(dim=0).cpu()

        total_samples += actual_batch

        # free GPU memory
        del resid_mid_cache, resid_post_cache, input_ids, attention_mask
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # compute population mean
    result = {}
    for key, s in running_sum.items():
        result[key] = s / total_samples

    logger.info(
        "Extracted activations for %d prompts across %d layers and 2 sites",
        total_samples,
        num_layers,
    )

    # log prompt length stats
    if prompt_lengths:
        lengths = np.array(prompt_lengths)
        logger.debug(
            "Prompt token lengths: min=%d, max=%d, mean=%.1f, median=%.1f",
            int(lengths.min()),
            int(lengths.max()),
            float(lengths.mean()),
            float(np.median(lengths)),
        )

    return result, prompt_lengths


# candidate direction computation


def compute_candidates(
    abstain_means: Dict[Tuple[int, str], torch.Tensor],
    answer_means: Dict[Tuple[int, str], torch.Tensor],
    norm_floor: float = DEFAULT_NORM_FLOOR,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float], List[str]]:
    """Compute candidate refusal directions.

    candidate = mean_normed(abstain) - mean_normed(answer), then L2-normalised.

    Returns
    -------
    candidates : dict
        Key `"(layer, site)"` -> unit-norm direction tensor.

    raw_norms : dict
        Pre-normalisation L2 norms (useful for diagnostics).

    viable_keys : list
        Keys with raw norm above `norm_floor`.
    """
    candidates: Dict[str, torch.Tensor] = {}
    raw_norms: Dict[str, float] = {}
    viable_keys: List[str] = []

    all_keys = sorted(abstain_means.keys(), key=lambda k: (k[0], k[1]))

    for key in all_keys:
        # normalise each class mean to unit norm, then subtract
        a_normed = normalize(abstain_means[key].unsqueeze(0), dim=-1).squeeze(0)
        b_normed = normalize(answer_means[key].unsqueeze(0), dim=-1).squeeze(0)
        raw_dir = a_normed - b_normed

        raw_norm = raw_dir.norm().item()
        str_key = str(key)
        raw_norms[str_key] = raw_norm

        if raw_norm > norm_floor:
            candidates[str_key] = raw_dir / max(raw_norm, 1e-12)
            viable_keys.append(str_key)
        else:
            normed = raw_dir / max(raw_norm, 1e-12)
            # replace NaNs from degenerate zero-norm directions
            normed = torch.nan_to_num(normed, nan=0.0)
            candidates[str_key] = normed

    logger.info(
        "Candidate directions: %d total, %d viable (norm > %s)",
        len(all_keys),
        len(viable_keys),
        norm_floor,
    )
    return candidates, raw_norms, viable_keys


# u1 selection


def select_u1(
    candidates: Dict[str, torch.Tensor],
    viable_keys: List[str],
) -> Tuple[str, torch.Tensor, torch.Tensor]:
    """Select the best refusal direction u1.

    Uses mean cosine similarity: the candidate with the highest mean cosine
    with all other viable candidates is selected as u1.
    """
    if not viable_keys:
        raise ValueError(
            "No viable candidate directions found. "
            "Try lowering --norm_floor or using more data."
        )

    if len(viable_keys) == 1:
        key = viable_keys[0]
        logger.info("Only one viable candidate - selected u1 = %s", key)
        cos_matrix = torch.ones(1, 1)
        return key, candidates[key], cos_matrix

    # build cosine matrix between viable candidates
    vecs = torch.stack([candidates[k] for k in viable_keys])  # (N, D)
    cos_matrix = vecs @ vecs.T  # (N, N)

    # mean cosine with all OTHER candidates
    n = len(viable_keys)
    # zero out the diagonal (self-similarity = 1)
    mask = 1 - torch.eye(n)
    masked = cos_matrix * mask
    mean_cos = masked.sum(dim=1) / (n - 1)
    best_idx = mean_cos.argmax().item()

    u1_key = viable_keys[best_idx]
    logger.info("Selected u1 = %s (metric=mean_cosine)", u1_key)

    # log per-candidate mean cosine for diagnostics
    for i, key in enumerate(viable_keys):
        logger.debug("  candidate %s: mean_cos=%.4f", key, mean_cos[i].item())

    return u1_key, candidates[u1_key], cos_matrix


# PCA plane construction


def build_steering_plane(
    candidates: Dict[str, torch.Tensor],
    viable_keys: List[str],
    u1: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Construct the (u1, u2) steering plane.

    u2 is derived from PCA on the viable candidate directions projected into
    the orthogonal complement of u1.  This ensures u2 is orthogonal to u1 by
    construction and avoids arbitrary alignment thresholds.

    Returns
    -------
    u1 : Tensor of shape (D,) - unit norm

    u2 : Tensor of shape (D,) - unit norm, orthogonal to u1

    explained_variance : ndarray - PCA explained variance ratios
    """
    u1_np = u1.numpy()

    # special case: <=1 viable candidate
    if len(viable_keys) <= 1:
        raise ValueError(
            "Need at least 2 viable candidate directions to construct an Angular "
            f"steering plane, got {len(viable_keys)}."
        )

    # standard case: PCA in orthogonal complement of u1
    vecs = torch.stack([candidates[k] for k in viable_keys]).numpy()

    # project every candidate into u1's orthogonal complement:
    #   x_i^perp  =  x_i  -  (x_i · u1) u1
    projections_onto_u1 = vecs @ u1_np
    vecs_perp = vecs - projections_onto_u1[:, None] * u1_np

    # PCA on the perpendicular components
    pca = PCA().fit(vecs_perp)

    # first PC is our u2 (already orthogonal to u1 by construction)
    u2 = pca.components_[0].copy()
    u2_norm = np.linalg.norm(u2)
    if u2_norm < 1e-10:
        raise ValueError(
            "Degenerate PCA while constructing u2: first orthogonal-complement "
            f"component has near-zero norm ({u2_norm:.2e})."
        )
    u2 /= u2_norm

    logger.info(
        "u2 from PCA component 0 in orthogonal complement of u1 "
        "(explained variance ratio: %.4f)",
        pca.explained_variance_ratio_[0],
    )

    u2_tensor = torch.tensor(u2, dtype=torch.float32)

    # verify orthogonality
    dot = float(u1 @ u2_tensor)
    assert abs(dot) < 1e-5, f"u1 · u2 = {dot}, not orthogonal!"
    logger.info("u1 · u2 = %.6f (should be ~0)", dot)

    return u1, u2_tensor, pca.explained_variance_ratio_


# notebook-format steering config (optional)


def build_notebook_steering_config(
    model,
    u1: torch.Tensor,
    u2: torch.Tensor,
    num_layers: int,
    model_name: str,
) -> dict:
    """Build a steering config dict in the Angular notebook's format.

    Keys are module names like `model.layers.{idx}.post_attention_layernorm`.
    Each value contains `first_direction`, `second_direction`, `mode`.

    Module targeting follows the notebook convention:
    - Non-Gemma: `input_layernorm` + `post_attention_layernorm`
    - Gemma: `post_attention_layernorm` + `post_feedforward_layernorm`

    `input_layernorm` at layer *i+1* is equivalent to `resid_post` at
    layer *i*, so for the last layer it is skipped.
    """
    # detect layernorm module names - match the notebook's convention
    if "gemma" in model_name.lower():
        layernorm_modules = [
            "post_attention_layernorm",
            "post_feedforward_layernorm",
        ]
    else:
        layernorm_modules = ["input_layernorm", "post_attention_layernorm"]

    u1_np = u1.numpy()
    u2_np = u2.numpy()

    config = {}
    for layer_idx in range(num_layers):
        for module in layernorm_modules:
            # input_layernorm at layer i+1 is resid_post at layer i
            if module == "input_layernorm":
                if layer_idx < num_layers - 1:
                    module_name = f"model.layers.{layer_idx + 1}.{module}"
                else:
                    continue  # skip - no next layer
            else:
                module_name = f"model.layers.{layer_idx}.{module}"

            config[module_name] = {
                "mode": "rotate_to",
                "first_direction": u1_np.copy(),
                "second_direction": u2_np.copy(),
            }

    return config


# main extraction flow


def extract_angular_vectors(args: argparse.Namespace) -> None:
    """Main entry point: load data, extract activations, compute plane, save."""

    # resolve device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    logger.info("Device: %s", device)

    # load data
    exclude_tasks = None
    if args.exclude_tasks:
        exclude_tasks = {t.strip() for t in args.exclude_tasks.split(",")}

    abstain_prompts, answer_prompts = load_abstention_dataset(
        args.data_path,
        max_samples=args.max_samples,
        exclude_tasks=exclude_tasks,
        seed=args.seed,
        dedupe=args.dedupe,
        stratified=args.stratified,
    )

    # load model
    logger.info("Loading model: %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # ensure left-padding for batched prompt extraction
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            raise ValueError("No pad_token or eos_token in tokenizer")

    # choose dtype by device capability
    if device.type == "cuda":
        model_dtype = torch.bfloat16
    elif device.type == "mps":
        model_dtype = torch.float16
    else:
        model_dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=model_dtype,
    )
    model.to(device)
    model.eval()
    logger.info("Model loaded on %s (dtype=%s)", device, model_dtype)

    # template suffix detection
    suffix_strs, num_suffix_tokens = get_template_suffix_tokens(
        tokenizer, args.use_system_prompt
    )

    # extract activations
    logger.info("Extracting abstain-class activations...")
    abstain_means, abstain_lengths = extract_activations(
        model,
        tokenizer,
        abstain_prompts,
        use_system_prompt=args.use_system_prompt,
        num_suffix_tokens=num_suffix_tokens,
        batch_size=args.batch_size,
        device=device,
        suffix_pool=args.suffix_pool,
    )

    logger.info("Extracting answer-class activations...")
    answer_means, answer_lengths = extract_activations(
        model,
        tokenizer,
        answer_prompts,
        use_system_prompt=args.use_system_prompt,
        num_suffix_tokens=num_suffix_tokens,
        batch_size=args.batch_size,
        device=device,
        suffix_pool=args.suffix_pool,
    )

    # candidate directions
    candidates, raw_norms, viable_keys = compute_candidates(
        abstain_means,
        answer_means,
        norm_floor=args.norm_floor,
    )

    # select u1
    u1_key, u1, cos_matrix = select_u1(
        candidates,
        viable_keys,
    )

    # class-separation sanity check: project class means onto u1
    try:
        u1_key_tuple = ast.literal_eval(u1_key)
        a_mean = normalize(abstain_means[u1_key_tuple].unsqueeze(0), dim=-1).squeeze(0)
        b_mean = normalize(answer_means[u1_key_tuple].unsqueeze(0), dim=-1).squeeze(0)
        a_proj = float(a_mean @ u1)
        b_proj = float(b_mean @ u1)
        logger.info(
            "Class separation on u1: abstain projection = %.4f, "
            "answer projection = %.4f (delta = %.4f)",
            a_proj,
            b_proj,
            a_proj - b_proj,
        )
        if a_proj < b_proj:
            logger.warning(
                "⚠ Unexpected sign: answer projects higher than abstain onto u1. "
                "The direction sign convention may be inverted."
            )
    except (ValueError, KeyError) as e:
        logger.debug("Could not compute class-separation check: %s", e)

    # build steering plane
    u1, u2, pca_variance = build_steering_plane(
        candidates,
        viable_keys,
        u1,
    )

    # --- NaN safety checks ---
    for name, tensor in [("u1", u1), ("u2", u2)]:
        assert not torch.isnan(tensor).any(), f"NaN detected in {name}!"
    assert not torch.isnan(cos_matrix).any(), "NaN detected in cosine matrix!"
    for k in viable_keys:
        assert not torch.isnan(candidates[k]).any(), (
            f"NaN detected in viable candidate {k}!"
        )
    # Non-viable candidates may have been zeroed from degenerate norms - that is expected
    n_non_viable_nan = sum(
        1
        for k, v in candidates.items()
        if k not in viable_keys and torch.isnan(v).any()
    )
    if n_non_viable_nan > 0:
        logger.debug(
            "%d non-viable candidates had NaN (replaced with zeros)", n_non_viable_nan
        )
    logger.debug("NaN check passed: u1, u2, cosine matrix, viable candidates clean")

    # save
    out_dir = os.path.dirname(args.output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    num_layers = model.config.num_hidden_layers

    # compute per-candidate mean cosine for diagnostics
    mean_cosine_per_candidate = {}
    if len(viable_keys) > 1:
        vecs = torch.stack([candidates[k] for k in viable_keys])
        cos_mat = vecs @ vecs.T
        n = len(viable_keys)
        mask = 1 - torch.eye(n)
        mean_cos = (cos_mat * mask).sum(dim=1) / (n - 1)
        for i, key in enumerate(viable_keys):
            mean_cosine_per_candidate[key] = float(mean_cos[i])

    # prompt length stats
    all_lengths = abstain_lengths + answer_lengths
    length_stats = {}
    if all_lengths:
        lengths_arr = np.array(all_lengths)
        length_stats = {
            "min": int(lengths_arr.min()),
            "max": int(lengths_arr.max()),
            "mean": float(lengths_arr.mean()),
            "median": float(np.median(lengths_arr)),
        }

    save_dict = {
        "u1": u1.float(),
        "u2": u2.float(),
        "u1_key": u1_key,
        "candidates": {k: v.float() for k, v in candidates.items()},
        "candidate_norms_raw": raw_norms,
        "cosine_matrix": cos_matrix.float(),
        "viable_keys": viable_keys,
        "pca_explained_variance": torch.tensor(pca_variance)
        if len(pca_variance) > 0
        else torch.tensor([]),
        "metadata": {
            "model_name": args.model_name,
            "data_path": args.data_path,
            "n_abstain_prompts": len(abstain_prompts),
            "n_answer_prompts": len(answer_prompts),
            "num_suffix_tokens": num_suffix_tokens,
            "suffix_tokens": suffix_strs,
            "suffix_pool": args.suffix_pool,
            "norm_floor": args.norm_floor,
            "use_system_prompt": args.use_system_prompt,
            "num_layers": num_layers,
            "hidden_dim": model.config.hidden_size,
            "seed": args.seed,
            "dedupe": args.dedupe,
            "stratified": args.stratified,
            "model_dtype": str(model_dtype),
            "exclude_tasks": list(exclude_tasks) if exclude_tasks else [],
            "mean_cosine_per_candidate": mean_cosine_per_candidate,
            "prompt_length_stats": length_stats,
        },
    }

    torch.save(save_dict, args.output_path)
    logger.info("Saved Angular steering artifacts to %s", args.output_path)

    # optional: notebook-format steering config
    if args.save_notebook_config:
        config = build_notebook_steering_config(
            model,
            u1,
            u2,
            num_layers,
            args.model_name,
        )
        config_path = args.output_path.replace(".pt", "_steering_config.npy")
        np.save(config_path, config)
        logger.info("Saved notebook-format steering config to %s", config_path)

    # summary
    logger.info("Angular extraction complete for %s", args.model_name)
    logger.info("  u1 key:        %s", u1_key)
    logger.info("  u1 norm:       %.6f", u1.norm())
    logger.info("  u2 norm:       %.6f", u2.norm())
    logger.info("  u1 · u2:       %.6f", float(u1 @ u2))
    logger.info(
        "  Viable / total candidates: %d / %d", len(viable_keys), len(candidates)
    )
    if len(pca_variance) > 0:
        logger.info("  PCA var (top-3): %s", pca_variance[:3])
    logger.info("  Suffix pool:   %s", args.suffix_pool)
    logger.info("  Output: %s", args.output_path)


# CLI


def main():
    parser = argparse.ArgumentParser(
        description="Angular Activation Steering - Direction Extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="HuggingFace model name or path (e.g. Qwen/Qwen2.5-0.5B-Instruct)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to abstention_training_dataset.json",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the .pt output file",
    )
    parser.add_argument(
        "--use_system_prompt",
        action="store_true",
        help="Include system prompt in chat template",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=512,
        help="Max prompts per class (default: 512, matching Angular notebook)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for activation extraction (default: 16)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: 'auto', 'cuda', 'mps', or 'cpu' (default: auto)",
    )
    parser.add_argument(
        "--norm_floor",
        type=float,
        default=DEFAULT_NORM_FLOOR,
        help=f"Minimum raw norm for a candidate to be viable "
        f"(default: {DEFAULT_NORM_FLOOR})",
    )
    parser.add_argument(
        "--exclude_tasks",
        type=str,
        default=None,
        help="Comma-separated task types to exclude (e.g. 'stale,subjective')",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sub-sampling (default: 42)",
    )
    parser.add_argument(
        "--save_notebook_config",
        action="store_true",
        help="Also save a steering_config.npy in the Angular notebook format "
        "(for use with the Angular vLLM fork)",
    )
    parser.add_argument(
        "--no_dedupe",
        action="store_true",
        help="Disable prompt deduplication by (question, should_abstain). "
        "By default, dedup is ON to avoid counting identical prompts "
        "multiple times.",
    )
    parser.add_argument(
        "--stratified",
        action="store_true",
        help="Use task-stratified subsampling instead of uniform random. "
        "Useful when task distribution is very skewed.",
    )
    parser.add_argument(
        "--suffix_pool",
        type=str,
        choices=["last", "mean"],
        default="last",
        help="How to pool over template suffix tokens. "
        "'last' uses only the final token position (closer to the notebook). "
        "'mean' averages over all suffix positions. (default: last)",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING"],
        default="INFO",
        help="Logging verbosity level (default: INFO)",
    )

    args = parser.parse_args()
    # derive dedupe boolean from --no_dedupe flag
    args.dedupe = not args.no_dedupe

    # configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    extract_angular_vectors(args)


if __name__ == "__main__":
    main()
