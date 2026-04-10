"""Angular steering: extract refusal directions and a (u1, u2) plane from prompts.

The procedure follows Vu and Nguyen (2025), adapted here to our abstention
dataset. Saved tensors and optional ``_steering_config.npy`` match the authors'
released vLLM v1 steering configuration and runtime interface.

Vu, H.M. and Nguyen, T.M., 2025. Angular steering: Behavior control via rotation
in activation space. arXiv:2510.26243.

Steps:

1. Split prompts by ``should_abstain``.
2. Run the chat-templated, left-padded forward pass without generating tokens.
3. At each layer, read ``resid_mid`` (post-attention LayerNorm input) and
   ``resid_post`` (decoder block output).
4. Pool activations at the template suffix.
5. Build class-difference candidates, select ``u1`` by mean pairwise cosine,
   and set ``u2`` from PCA on candidates.
6. Save a ``.pt`` file and optionally a notebook-format steering config.

Examples:

    python angular/extract_angular.py \\
        --model_name Qwen/Qwen2.5-0.5B-Instruct \\
        --data_path data/abstention_training_dataset.json \\
        --output_path data/angular_vectors/Qwen2_5_0_5B/angular_steering.pt \\
        --use_system_prompt \\
        --max_samples 32 \\
        --batch_size 4
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

# candidates at or below this raw (pre-normalisation) L2 norm are excluded from
# u1 selection and PCA. The reference notebook used 0.
DEFAULT_NORM_FLOOR = 0.0


# data loading


def load_abstention_dataset(
    path: str,
    max_samples: Optional[int] = None,
    exclude_tasks: Optional[set] = None,
    seed: int = 42,
    dedupe: bool = True,
    stratified: bool = False,
) -> Tuple[List[str], List[str]]:
    """Load prompts and split into should-abstain and should-answer lists.

    Args:
        path: Path to the JSON dataset.
        max_samples: Maximum rows per class after optional deduplication. Use
            ``None`` to keep all rows.
        exclude_tasks: If set, drop any row whose ``task`` is in this set.
        seed: RNG seed for subsampling.
        dedupe: If True, keep one row per (question, ``should_abstain``) pair so
            paired responses do not duplicate the same question text.
        stratified: If True and ``max_samples`` is set, subsample by task
            proportionally instead of uniformly at random.

    Returns:
        A pair ``(abstain_prompts, answer_prompts)`` of disjoint question lists.
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
    """Infer chat-template suffix tokens after the variable user message.

    Tokenizes two one-character user messages and keeps the shared tail after
    the last position where they differ. That tail is the generation prompt and
    assistant prefix region.

    Returns:
        ``suffix_token_strs`` (for logging) and ``num_suffix_tokens`` (for pooling).
    """

    def _tokenize(text: str) -> torch.Tensor:
        if use_system_prompt:
            msgs = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ]
        else:
            msgs = [{"role": "user", "content": text}]
        # use tokenize=True for a single-pass tokenization that preserves
        # token boundaries (consistent with the CAA script)
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

    # scan from the end to find the first position where tokens differ
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
    """Apply the chat template with batch padding.

    The tokenizer should use ``padding_side == "left"`` so batched prompts align
    on the right.

    Returns:
        ``input_ids`` and ``attention_mask``, each with shape ``(batch, seq)``.
        Mask values are 0 on padding tokens.
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


def _find_first_decoder_layer(model) -> Tuple[str, object]:
    """Return `(module_path, module)` for the first decoder block."""
    for name, module in model.named_modules():
        # common patterns: model.layers.0, model.model.layers.0
        if re.search(r"layers[.\[]0[.\]]?$", name) or name.endswith("layers.0"):
            return name, module

    raise RuntimeError(
        "Cannot find decoder layers. Supported architectures: "
        "Llama, Qwen, Gemma, Mistral"
    )


def _detect_mid_layernorm_pattern(model) -> str:
    """Infer the post-attention LayerNorm path pattern for ``resid_mid`` hooks.

    Returns:
        Format string with ``{layer_idx}`` for ``model.get_submodule``.
    """
    base_path, layer_module = _find_first_decoder_layer(model)

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
    """Infer one decoder block path pattern for ``resid_post`` forward hooks.

    Returns:
        Format string with ``{layer_idx}``.
    """
    name, _ = _find_first_decoder_layer(model)
    pattern = name.replace(".0", ".{layer_idx}")
    logger.info("Detected decoder layer pattern: %s", pattern)
    return pattern


def _detect_steering_target_specs(model, model_name: str) -> List[Tuple[str, int]]:
    """Module names and layer offsets for exporting Angular steering targets.

    Each entry is ``(module_name, layer_offset)``. Offset 0 attaches to the same
    layer index. Offset 1 attaches to the next layer, which on Llama-like
    stacks can map ``resid_post`` through the next layer's ``input_layernorm``.
    """
    _, layer_module = _find_first_decoder_layer(model)
    child_names = {n for n, _ in layer_module.named_children()}
    is_gemma_like = (
        "gemma" in model_name.lower()
        or "pre_feedforward_layernorm" in child_names
        or "post_feedforward_layernorm" in child_names
    )

    specs: List[Tuple[str, int]] = []

    if not is_gemma_like and "input_layernorm" in child_names:
        specs.append(("input_layernorm", 1))

    if "post_attention_layernorm" in child_names:
        specs.append(("post_attention_layernorm", 0))

    # Gemma variants use a feedforward LayerNorm name instead of the
    # Llama/Qwen/Mistral `input_layernorm` + `post_attention_layernorm` pair
    if is_gemma_like:
        if "post_feedforward_layernorm" in child_names:
            specs.append(("post_feedforward_layernorm", 0))
        elif "pre_feedforward_layernorm" in child_names:
            specs.append(("pre_feedforward_layernorm", 0))

    if not specs:
        raise RuntimeError(
            "Could not identify steering target modules from first decoder layer. "
            f"Children: {sorted(child_names)}"
        )

    logger.info("Detected steering target modules: %s", specs)
    return specs


def _pool_suffix(
    acts: torch.Tensor,
    num_suffix_tokens: int,
    suffix_pool: str,
    do_normalize: bool = True,
) -> torch.Tensor:
    """Pool activations over the template suffix.

    Args:
        acts: Activations with shape ``(batch, seq, hidden)``.
        num_suffix_tokens: Number of positions at the end of the sequence that
            belong to the suffix window.
        suffix_pool: ``last`` keeps only the final position. ``mean`` averages
            over the last ``num_suffix_tokens`` positions.
        do_normalize: If True, L2-normalize each token vector before pooling.

    Returns:
        Tensor of shape ``(batch, hidden)``.
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
) -> Tuple[Dict[Tuple[int, str], torch.Tensor], List[int]]:
    """Mean pooled activations per (layer, site) over all prompts.

    Accumulates running sums over batches. Registers pre-forward hooks on the
    post-attention LayerNorm for ``resid_mid`` and forward hooks on decoder
    blocks for ``resid_post``. Pools suffix positions using ``suffix_pool`` and
    optionally L2-normalizes per token before pooling.

    Returns:
        A tuple ``(means, prompt_token_lengths)``.

        ``means`` maps each key ``(layer_idx, "resid_mid"|"resid_post")`` to a
        mean vector of shape ``(hidden,)``.

        ``prompt_token_lengths`` lists the non-padding token count for each
        prompt.
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
    """Class-difference directions from unit-normalized abstain minus answer means.

    Returns:
        A tuple ``(candidates, raw_norms, viable_keys)``.

        Dictionary keys are strings ``"(layer, site)"``. The list
        ``viable_keys`` names candidates whose raw norm exceeds ``norm_floor``.
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
    """Select ``u1`` as the viable candidate with the highest mean cosine to all others."""
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
    """Set ``u2`` to the first PCA axis on viable candidates. Leaves ``u1`` fixed.

    The steering runtime orthogonalizes ``u2`` against ``u1`` when it builds the
    basis.

    Returns:
        ``u1``, ``u2``, and ``explained_variance_ratio``. Vectors ``u1`` and
        ``u2`` are unit norm.
    """
    # special case: <=1 viable candidate
    if len(viable_keys) <= 1:
        raise ValueError(
            "Need at least 2 viable candidate directions to construct an Angular "
            f"steering plane, got {len(viable_keys)}."
        )

    # notebook-faithful: PCA on the viable candidate directions directly
    vecs = torch.stack([candidates[k] for k in viable_keys]).numpy()
    pca = PCA().fit(vecs)

    # first PC is saved raw to match the notebook
    # the runtime orthogonalizes it against u1 when constructing the steering basis
    u2 = pca.components_[0].copy()
    u2_norm = np.linalg.norm(u2)
    if u2_norm < 1e-10:
        raise ValueError(
            "Degenerate PCA while constructing u2: first component has near-zero "
            f"norm ({u2_norm:.2e})."
        )
    u2 /= u2_norm

    logger.info(
        "u2 from raw PCA component 0 (not orthogonalized, explained variance "
        "ratio %.4f)",
        pca.explained_variance_ratio_[0],
    )

    u2_tensor = torch.tensor(u2, dtype=torch.float32)

    dot = float(u1 @ u2_tensor)
    logger.info("u1 · u2 = %.6f before runtime orthogonalization", dot)

    return u1, u2_tensor, pca.explained_variance_ratio_


# notebook-format steering config (optional)


def build_notebook_steering_config(
    model,
    u1: torch.Tensor,
    u2: torch.Tensor,
    num_layers: int,
    model_name: str,
) -> dict:
    """Build a per-module map for Angular and vLLM steering.

    Each value uses ``mode="rotate_to"`` with ``first_direction`` and
    ``second_direction``. Keys use detected submodule paths instead of a fixed
    ``model.layers.*`` prefix. Module choices follow the published notebook
    layout for Llama-like versus Gemma stacks. Entries that would use a
    next-layer ``input_layernorm`` beyond the last layer are omitted.
    """
    decoder_pattern = _detect_decoder_layer_pattern(model)
    target_specs = _detect_steering_target_specs(model, model_name)

    u1_np = u1.detach().cpu().numpy()
    u2_np = u2.detach().cpu().numpy()

    config = {}
    for layer_idx in range(num_layers):
        for module_name, layer_offset in target_specs:
            target_layer_idx = layer_idx + layer_offset
            if target_layer_idx >= num_layers:
                continue

            full_module_name = (
                decoder_pattern.format(layer_idx=target_layer_idx) + f".{module_name}"
            )

            config[full_module_name] = {
                "mode": "rotate_to",
                "first_direction": u1_np.copy(),
                "second_direction": u2_np.copy(),
            }

    return config


# main extraction flow


def extract_angular_vectors(args: argparse.Namespace) -> None:
    """Run the full extraction pipeline. Writes ``.pt`` and optionally ``_steering_config.npy``."""

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

    # NaN checks on exported tensors
    for name, tensor in [("u1", u1), ("u2", u2)]:
        assert not torch.isnan(tensor).any(), f"NaN detected in {name}!"
    assert not torch.isnan(cos_matrix).any(), "NaN detected in cosine matrix!"
    for k in viable_keys:
        assert not torch.isnan(candidates[k]).any(), (
            f"NaN detected in viable candidate {k}!"
        )
    # non-viable candidates may have been zeroed from degenerate norms - that is expected
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
            "u2_construction": "raw_pca_component_0",
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
        description=(
            "Extract Angular steering directions (Vu and Nguyen, 2025). "
            "See the module docstring for the full citation."
        ),
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
        "(for use with the Angular runtimes, including vLLM v1 steering)",
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
