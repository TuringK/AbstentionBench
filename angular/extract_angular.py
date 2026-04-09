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
4. Average activations over the template suffix tokens - the shared tail
   tokens that follow the variable user content (e.g. `<|im_start|>assistant\n`).
5. Per-token L2-normalize, then mean over suffix positions -> one vector per
   (sample, layer, site).
6. Candidate direction = mean_normed(abstain) - mean_normed(answer) per
   (layer, site).  L2-normalize each candidate.
7. Select best direction u1 (highest mean cosine with other candidates).
8. Build orthogonal second basis u2 via PCA on all viable candidates +
   Gram-Schmidt orthogonalisation against u1.
9. Save `{u1, u2, candidates, metadata}` as a `.pt` file.

Usage
-----
Local (Mac / CPU / MPS):

    python angular/extract_angular.py \
        --model_name Qwen/Qwen2.5-0.5B-Instruct \
        --data_path data/abstention_training_dataset.json \
        --output_path data/angular_vectors/Qwen2_5_0_5B/angular_steering.pt \
        --use_system_prompt \
        --max_samples 32 \
        --batch_size 4

HPC (CUDA):

    python angular/extract_angular.py \
        --model_name Qwen/Qwen2.5-7B-Instruct \ 
        --data_path data/abstention_training_dataset.json \
        --output_path data/angular_vectors/Qwen2_5_7B/angular_steering.pt \
        --use_system_prompt \
        --max_samples 512 \
        --batch_size 16
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA
from torch.nn.functional import normalize
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from recipe.system_prompt import SYSTEM_PROMPT

# candidates with a raw (pre-normalisation) L2 norm below this threshold are
# considered noise and excluded from u1 selection and PCA.
DEFAULT_NORM_FLOOR = 0.01

# minimum cosine between u2 and u1 below which we accept u2 as sufficiently
# orthogonal.  cos(85°) ≈ 0.087
U2_ALIGNMENT_THRESHOLD = 0.10


# data loading

def load_abstention_dataset(
    path: str,
    max_samples: Optional[int] = None,
    exclude_tasks: Optional[set] = None,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """Load and partition prompts by `should_abstain`.

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
        print(f"Excluded tasks {exclude_tasks}: {before} -> {len(data)} entries")

    abstain_prompts = [d["question"] for d in data if d["should_abstain"]]
    answer_prompts = [d["question"] for d in data if not d["should_abstain"]]

    print(
        f"Dataset: {len(abstain_prompts)} should-abstain, "
        f"{len(answer_prompts)} should-answer prompts"
    )

    # sub-sample if requested (deterministic)
    if max_samples is not None:
        rng = np.random.default_rng(seed)
        if len(abstain_prompts) > max_samples:
            idx = rng.choice(len(abstain_prompts), max_samples, replace=False)
            abstain_prompts = [abstain_prompts[i] for i in sorted(idx)]
        if len(answer_prompts) > max_samples:
            idx = rng.choice(len(answer_prompts), max_samples, replace=False)
            answer_prompts = [answer_prompts[i] for i in sorted(idx)]
        print(
            f"Sub-sampled to {len(abstain_prompts)} abstain, "
            f"{len(answer_prompts)} answer prompts"
        )

    return abstain_prompts, answer_prompts


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
        chat_str = tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
        )
        ids = tokenizer.encode(chat_str, add_special_tokens=False)
        return torch.tensor(ids, dtype=torch.long)

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

    print(f"Template suffix tokens ({num_suffix}): {suffix_strs}")
    return suffix_strs, num_suffix


# prompt tokenization


def prompts_to_chat_tokens(
    tokenizer: AutoTokenizer,
    prompts: List[str],
    use_system_prompt: bool,
) -> torch.Tensor:
    """Tokenize a list of prompts with the chat template, left-padded."""
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
    # newer transformers returns BatchEncoding; extract the tensor
    if hasattr(result, "input_ids"):
        return result.input_ids
    return result


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

    print(f"Detected resid_mid layernorm pattern: {pattern}")
    return pattern


def extract_activations(
    model,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    use_system_prompt: bool,
    num_suffix_tokens: int,
    batch_size: int,
    device: torch.device,
    normalize_acts: bool = True,
) -> Dict[Tuple[int, str], torch.Tensor]:
    """Extract mean activations at (layer, site) for a set of prompts.

    Processes prompts in batches.  For each batch:
    - Registers pre-forward hooks on LayerNorm modules to capture `resid_mid`.
    - Uses `output_hidden_states=True` to capture `resid_post`.
    - Extracts the last `num_suffix_tokens` positions from each.
    - L2-normalizes per token (if `normalize_acts`), then averages.

    Returns a dict mapping `(layer_idx, "resid_mid"|"resid_post")` to a
    tensor of shape `(hidden_dim,)` - the population mean.

    Memory-efficient: accumulates running sums instead of storing per-sample
    activations.
    """
    num_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    ln_pattern = _detect_mid_layernorm_pattern(model)

    # running accumulators: (layer, site) -> running sum of (normalised) vectors
    running_sum: Dict[Tuple[int, str], torch.Tensor] = {}
    for layer_idx in range(num_layers):
        for site in ("resid_mid", "resid_post"):
            running_sum[(layer_idx, site)] = torch.zeros(
                hidden_dim, dtype=torch.float32
            )
    total_samples = 0

    for batch_start in tqdm(
        range(0, len(prompts), batch_size),
        desc="Extracting activations",
    ):
        batch_prompts = prompts[batch_start : batch_start + batch_size]

        input_ids = prompts_to_chat_tokens(
            tokenizer, batch_prompts, use_system_prompt
        ).to(device)

        # register hooks for resid_mid
        resid_mid_cache: Dict[int, torch.Tensor] = {}
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

        # forward pass
        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                output_hidden_states=True,
            )

        # remove hooks
        for h in hooks:
            h.remove()

        actual_batch = input_ids.shape[0]

        # process resid_mid
        for layer_idx in range(num_layers):
            raw = resid_mid_cache[layer_idx]  # (batch, seq, hidden)
            suffix = raw[:, -num_suffix_tokens:, :].float()  # cast small slice

            if normalize_acts:
                suffix = normalize(suffix, dim=-1)

            # mean over suffix tokens -> (batch, hidden)
            mean_per_sample = suffix.mean(dim=1)
            # accumulate batch sum
            running_sum[(layer_idx, "resid_mid")] += mean_per_sample.sum(dim=0).cpu()

        # process resid_post: hidden_states[layer_idx + 1] = output of block
        for layer_idx in range(num_layers):
            hs = outputs.hidden_states[layer_idx + 1]  # (batch, seq, hidden)
            suffix = hs[:, -num_suffix_tokens:, :].float()

            if normalize_acts:
                suffix = normalize(suffix, dim=-1)

            mean_per_sample = suffix.mean(dim=1)
            running_sum[(layer_idx, "resid_post")] += mean_per_sample.sum(dim=0).cpu()

        total_samples += actual_batch

        # free GPU memory
        del outputs, resid_mid_cache, input_ids
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # compute population mean
    result = {}
    for key, s in running_sum.items():
        result[key] = s / total_samples

    print(
        f"Extracted activations for {total_samples} prompts across "
        f"{num_layers} layers × 2 sites"
    )
    return result


# candidate direction computation


def compute_candidates(
    abstain_means: Dict[Tuple[int, str], torch.Tensor],
    answer_means: Dict[Tuple[int, str], torch.Tensor],
    norm_floor: float = DEFAULT_NORM_FLOOR,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float], List[str]]:
    """Compute candidate refusal directions.

    candidate = mean_normed(abstain) − mean_normed(answer), then L2-normalised.

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
            candidates[str_key] = raw_dir / max(raw_norm, 1e-12)

    print(
        f"Candidate directions: {len(all_keys)} total, "
        f"{len(viable_keys)} viable (norm > {norm_floor})"
    )
    return candidates, raw_norms, viable_keys


# u1 selection


def select_u1(
    candidates: Dict[str, torch.Tensor],
    viable_keys: List[str],
    metric: str = "mean_cosine",
) -> Tuple[str, torch.Tensor, torch.Tensor]:
    """Select the best refusal direction u1.

    Parameters
    ----------
    metric : str
        `"mean_cosine"` - highest mean cosine with all other viable candidates.
        `"norm"` - highest raw direction norm (before unit normalisation).
    """
    if not viable_keys:
        raise ValueError(
            "No viable candidate directions found. "
            "Try lowering --norm_floor or using more data."
        )

    if len(viable_keys) == 1:
        key = viable_keys[0]
        print(f"Only one viable candidate - selected u1 = {key}")
        return key, candidates[key]

    # build cosine matrix between viable candidates
    vecs = torch.stack([candidates[k] for k in viable_keys])  # (N, D)
    cos_matrix = vecs @ vecs.T  # (N, N)

    if metric == "mean_cosine":
        # mean cosine with all OTHER candidates
        n = len(viable_keys)
        # zero out the diagonal (self-similarity = 1)
        mask = 1 - torch.eye(n)
        masked = cos_matrix * mask
        mean_cos = masked.sum(dim=1) / (n - 1)
        best_idx = mean_cos.argmax().item()
    elif metric == "norm":
        # already unit-normalised, but we use position in viable_keys
        # as a proxy - caller should pass raw_norms for a proper norm metric
        # for simplicity, fall back to mean_cosine
        mean_cos = (cos_matrix * (1 - torch.eye(len(viable_keys)))).sum(dim=1)
        best_idx = mean_cos.argmax().item()
    else:
        raise ValueError(f"Unknown selection metric: {metric}")

    u1_key = viable_keys[best_idx]
    print(f"Selected u1 = {u1_key} (metric={metric})")
    return u1_key, candidates[u1_key], cos_matrix


# PCA plane construction


def build_steering_plane(
    candidates: Dict[str, torch.Tensor],
    viable_keys: List[str],
    u1: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Construct the (u1, u2) steering plane.

    u2 is derived from PCA on all viable candidate directions,
    orthogonalised against u1 via Gram-Schmidt.

    Returns
    -------
    u1 : Tensor of shape (D,) - unit norm
    u2 : Tensor of shape (D,) - unit norm, orthogonal to u1
    explained_variance : ndarray - PCA explained variance ratios
    """
    vecs = torch.stack([candidates[k] for k in viable_keys]).numpy()

    pca = PCA().fit(vecs)

    u1_np = u1.numpy()

    # try PCA components in order until we find one sufficiently orthogonal to u1
    u2 = None
    for i, component in enumerate(pca.components_):
        # orthogonalise via Gram-Schmidt
        proj = float(component @ u1_np) * u1_np
        ortho = component - proj
        ortho_norm = np.linalg.norm(ortho)

        if ortho_norm > U2_ALIGNMENT_THRESHOLD:
            u2 = ortho / ortho_norm
            print(
                f"u2 from PCA component {i} "
                f"(orthogonal residual norm = {ortho_norm:.4f})"
            )
            break

    if u2 is None:
        # fallback: random orthogonal vector
        print(
            "Warning: All PCA components too aligned with u1. "
            "Using random orthogonal vector."
        )
        rng = np.random.default_rng(42)
        rand_vec = rng.standard_normal(u1_np.shape)
        proj = float(rand_vec @ u1_np) * u1_np
        ortho = rand_vec - proj
        u2 = ortho / np.linalg.norm(ortho)

    u2_tensor = torch.tensor(u2, dtype=torch.float32)

    # verify orthogonality
    dot = float(u1 @ u2_tensor)
    assert abs(dot) < 1e-5, f"u1 · u2 = {dot}, not orthogonal!"
    print(f"u1 · u2 = {dot:.6f} (should be ~0)")

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
    """
    # detect layernorm module names
    layernorm_modules = ["post_attention_layernorm"]
    if "gemma" in model_name.lower():
        layernorm_modules = [
            "post_attention_layernorm",
            "post_feedforward_layernorm",
        ]

    u1_np = u1.numpy()
    u2_np = u2.numpy()

    config = {}
    for layer_idx in range(num_layers):
        for module in layernorm_modules:
            if module == "input_layernorm" and layer_idx < num_layers - 1:
                module_name = f"model.layers.{layer_idx + 1}.{module}"
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
    print(f"Device: {device}")

    # load data
    exclude_tasks = None
    if args.exclude_tasks:
        exclude_tasks = {t.strip() for t in args.exclude_tasks.split(",")}

    abstain_prompts, answer_prompts = load_abstention_dataset(
        args.data_path,
        max_samples=args.max_samples,
        exclude_tasks=exclude_tasks,
        seed=args.seed,
    )

    # load model
    print(f"\nLoading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # ensure left-padding for batched prompt extraction
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            raise ValueError("No pad_token or eos_token in tokenizer")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16,
        device_map=device if device.type != "mps" else "auto",
    )
    model.eval()
    actual_device = next(model.parameters()).device
    print(f"Model loaded on {actual_device}")

    # template suffix detection
    suffix_strs, num_suffix_tokens = get_template_suffix_tokens(
        tokenizer, args.use_system_prompt
    )

    # extract activations
    print()
    abstain_means = extract_activations(
        model,
        tokenizer,
        abstain_prompts,
        use_system_prompt=args.use_system_prompt,
        num_suffix_tokens=num_suffix_tokens,
        batch_size=args.batch_size,
        device=actual_device,
    )

    print()
    answer_means = extract_activations(
        model,
        tokenizer,
        answer_prompts,
        use_system_prompt=args.use_system_prompt,
        num_suffix_tokens=num_suffix_tokens,
        batch_size=args.batch_size,
        device=actual_device,
    )

    # candidate directions
    print()
    candidates, raw_norms, viable_keys = compute_candidates(
        abstain_means,
        answer_means,
        norm_floor=args.norm_floor,
    )

    # select u1
    print()
    u1_key, u1, cos_matrix = select_u1(
        candidates,
        viable_keys,
        metric=args.selection_metric,
    )

    # build steering plane
    print()
    u1, u2, pca_variance = build_steering_plane(
        candidates,
        viable_keys,
        u1,
    )

    # sign sanity check
    # verify u1 projects abstain means higher than answer means on average
    u1_key_tuple = eval(u1_key)  # "(layer, 'site')" -> (layer, 'site')
    abstain_proj = float(abstain_means[u1_key_tuple] @ u1)
    answer_proj = float(answer_means[u1_key_tuple] @ u1)
    if abstain_proj < answer_proj:
        print(
            f"Warning: u1 sign flip needed (abstain proj={abstain_proj:.4f} "
            f"< answer proj={answer_proj:.4f}). Flipping u1."
        )
        u1 = -u1
    else:
        print(
            f"Sign check passed: abstain proj={abstain_proj:.4f} "
            f"> answer proj={answer_proj:.4f}"
        )

    # save
    out_dir = os.path.dirname(args.output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    num_layers = model.config.num_hidden_layers

    save_dict = {
        "u1": u1.float(),
        "u2": u2.float(),
        "u1_key": u1_key,
        "candidates": {k: v.float() for k, v in candidates.items()},
        "candidate_norms_raw": raw_norms,
        "cosine_matrix": cos_matrix.float(),
        "viable_keys": viable_keys,
        "pca_explained_variance": torch.tensor(pca_variance),
        "metadata": {
            "model_name": args.model_name,
            "data_path": args.data_path,
            "n_abstain_prompts": len(abstain_prompts),
            "n_answer_prompts": len(answer_prompts),
            "num_suffix_tokens": num_suffix_tokens,
            "suffix_tokens": suffix_strs,
            "selection_metric": args.selection_metric,
            "norm_floor": args.norm_floor,
            "use_system_prompt": args.use_system_prompt,
            "num_layers": num_layers,
            "hidden_dim": model.config.hidden_size,
        },
    }

    torch.save(save_dict, args.output_path)
    print(f"\nSaved Angular steering artifacts to {args.output_path}")

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
        print(f"Saved notebook-format steering config to {config_path}")

    # summary
    print()
    print(f"Angular extraction complete for {args.model_name}")
    print(f"  u1 key:        {u1_key}")
    print(f"  u1 norm:       {u1.norm():.6f}")
    print(f"  u2 norm:       {u2.norm():.6f}")
    print(f"  u1 · u2:       {float(u1 @ u2):.6f}")
    print(f"  Viable / total candidates: {len(viable_keys)} / {len(candidates)}")
    print(f"  PCA var (top-3): {pca_variance[:3]}")
    print(f"  Output: {args.output_path}")
    print()


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
        "--selection_metric",
        type=str,
        default="mean_cosine",
        choices=["mean_cosine", "norm"],
        help="Metric for selecting u1 (default: mean_cosine)",
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

    args = parser.parse_args()
    extract_angular_vectors(args)


if __name__ == "__main__":
    main()
