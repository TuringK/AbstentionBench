import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import List

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from recipe.system_prompt import SYSTEM_PROMPT


def _resolve_data_format(data_path: str, data_format: str) -> str:
    """Resolve dataset format, inferring from the file extension when ``auto``.

    Args:
        data_path: Path to the dataset file.
        data_format: ``"auto"``, ``"csv"``, or ``"json"``. With ``"auto"``,
            ``.json`` / ``.jsonl`` map to JSON; other extensions map to CSV.

    Returns:
        Either ``"csv"`` or ``"json"``.
    """
    if data_format != "auto":
        return data_format

    suffix = Path(data_path).suffix.lower()
    if suffix in {".json", ".jsonl"}:
        return "json"
    return "csv"


def _load_csv_pairs(df: pd.DataFrame, excluded: set[str]) -> list[dict]:
    """Build abstain vs non-abstain pairs from CSV rows grouped by ``pair_id``.

    Expects two rows per ``pair_id``: one with ``did_abstain`` true and one false.

    Args:
        df: Dataframe with ``pair_id``, ``did_abstain``, ``question``, ``response``,
            and optionally ``scenario``.
        excluded: Scenario names to skip.

    Returns:
        Dicts with keys ``question``, ``abstain_response``, ``non_abstain_response``,
        and ``scenario``.
    """
    df = df.dropna(subset=["response"])
    pairs = []

    grouped = df.groupby("pair_id")
    for _, group in grouped:
        if len(group) != 2:
            continue

        abstain_row = group[group["did_abstain"]]
        non_abstain_row = group[~group["did_abstain"]]

        if len(abstain_row) == 1 and len(non_abstain_row) == 1:
            scenario = str(abstain_row.iloc[0].get("scenario", "unknown")).strip()
            if scenario in excluded:
                continue

            pairs.append(
                {
                    "question": str(abstain_row.iloc[0]["question"]),
                    "abstain_response": str(abstain_row.iloc[0]["response"]),
                    "non_abstain_response": str(non_abstain_row.iloc[0]["response"]),
                    "scenario": scenario,
                }
            )
    return pairs


def _load_json_pairs(data: list[dict], excluded: set[str]) -> list[dict]:
    """Build pairs from JSON examples using ``positive``, ``negative``, and ``should_abstain``.

    Maps ``positive`` / ``negative`` to abstain vs non-abstain text depending on
    ``should_abstain`` (see inline comment in code).

    Args:
        data: List of objects with ``question``, ``positive``, ``negative``,
            and ``should_abstain``. Scenario comes from ``task``, ``scenario``, or
            ``dataset``.
        excluded: Scenario names to skip.

    Returns:
        Same dict shape as ``_load_csv_pairs``.
    """
    pairs = []

    for row in data:
        question = row.get("question")
        positive = row.get("positive")
        negative = row.get("negative")
        should_abstain = row.get("should_abstain")

        if not isinstance(question, str) or not question.strip():
            continue
        if not isinstance(positive, str) or not positive.strip():
            continue
        if not isinstance(negative, str) or not negative.strip():
            continue
        if not isinstance(should_abstain, bool):
            continue

        # the JSON schema stores a "positive" answer as the desired behavior:
        # abstain when should_abstain=True, answer otherwise
        if should_abstain:
            abstain_response, non_abstain_response = positive, negative
        else:
            abstain_response, non_abstain_response = negative, positive

        scenario = str(
            row.get("task") or row.get("scenario") or row.get("dataset") or "unknown"
        ).strip()
        if scenario in excluded:
            continue

        pairs.append(
            {
                "question": question,
                "abstain_response": abstain_response,
                "non_abstain_response": non_abstain_response,
                "scenario": scenario,
            }
        )

    return pairs


def load_pairs(data_path: str, data_format: str, excluded: set[str]) -> list[dict]:
    """Load abstention contrast pairs from CSV or JSON.

    Args:
        data_path: Path to a ``.csv`` or ``.json`` / ``.jsonl`` file.
        data_format: ``"auto"`` (infer), ``"csv"``, or ``"json"``.
        excluded: Scenarios to exclude.

    Returns:
        List of pair dicts ready for extraction.

    Raises:
        ValueError: If JSON content is not a top-level list.
    """
    resolved_format = _resolve_data_format(data_path, data_format)
    print(f"Detected data format: {resolved_format}")

    if resolved_format == "csv":
        df = pd.read_csv(data_path)
        return _load_csv_pairs(df=df, excluded=excluded)

    with open(data_path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("JSON dataset must be a list of examples.")

    return _load_json_pairs(data=data, excluded=excluded)


def extract_vectors(args: argparse.Namespace) -> None:
    """Run CAA-style extraction: mean activation difference (abstain minus non-abstain).

    Loads the model, tokenizes each question with the chat template, averages
    the first ``response_tokens`` hidden states at ``layer_idx`` for each branch,
    aggregates pair-wise differences (optionally weighted by scenario), optionally
    L2-normalizes, and saves one ``.pt`` steering vector.

    Args:
        args: Parsed CLI namespace with ``model_name``, ``data_path``,
            ``data_format``, ``output_path``, ``layer_idx``, ``use_system_prompt``,
            ``max_pairs``, ``weighted``, ``exclude_scenarios``, ``response_tokens``,
            and ``normalize`` (see ``argparse`` definitions in ``__main__``).
    """
    print(f"Loading model: {args.model_name}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    device = next(model.parameters()).device

    print(f"Loading data from {args.data_path}")
    # parse excluded scenarios
    excluded = set()
    if args.exclude_scenarios:
        excluded = {s.strip() for s in args.exclude_scenarios.split(",")}
        print(f"Excluding scenarios: {excluded}")

    pairs = load_pairs(
        data_path=args.data_path, data_format=args.data_format, excluded=excluded
    )

    print(f"Found {len(pairs)} valid pairs (after exclusions).")

    # print scenario distribution
    scenario_counts = defaultdict(int)
    for p in pairs:
        scenario_counts[p["scenario"]] += 1

    print("\nScenario distribution:")
    for scenario, count in sorted(scenario_counts.items(), key=lambda x: -x[1]):
        pct = 100.0 * count / len(pairs) if pairs else 0
        print(f"  {scenario:30s} {count:5d}  ({pct:5.1f}%)")
    print()

    if args.max_pairs is not None and len(pairs) > args.max_pairs:
        print(f"Limiting to {args.max_pairs} pairs.")
        pairs = pairs[: args.max_pairs]

    # compute per-pair diff vectors
    diff_vectors_by_scenario = defaultdict(list)

    for pair in pairs:
        question = pair["question"]
        scenario = pair["scenario"]

        responses = {
            "abstain": pair["abstain_response"],
            "non_abstain": pair["non_abstain_response"],
        }

        # tokenize the prompt via apply_chat_template(tokenize=True) to get
        # exact token IDs. Previously we tokenized the prompt as a string and then
        # re-tokenized prompt+response together - BPE merges can cross the
        # concatenation boundary, shifting the response offset by 1-2 tokens
        prompt_ids = question_to_chat_ids(
            use_system_prompt=args.use_system_prompt,
            question=question,
            tokenizer=tokenizer,
            device=device,
        )
        prompt_len = prompt_ids.shape[1]

        activations = {}

        for label, response in responses.items():
            response_ids = tokenizer(
                response,
                add_special_tokens=False,
                return_tensors="pt",
            ).input_ids.to(device)

            if response_ids.shape[1] == 0:
                continue

            input_ids = torch.cat([prompt_ids, response_ids], dim=1)
            attention_mask = torch.ones_like(input_ids, device=device)

            with torch.inference_mode():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )

            # cast to float32 before averaging. The model runs in bf16,
            # but accumulating many activations in reduced precision risks
            # losing mantissa bits. This is cheap and eliminates the concern
            hidden_state = outputs.hidden_states[args.layer_idx + 1].float()

            # extract only the response tokens (exclude prompt)
            response_acts = hidden_state[:, prompt_len:, :]

            # use only the first N tokens, or the length of the response, whichever is shorter
            num_tokens = response_acts.shape[1]
            slice_len = min(num_tokens, args.response_tokens)

            if slice_len == 0:
                continue

            mean_act = response_acts[:, :slice_len, :].mean(dim=1).squeeze()
            activations[label] = mean_act

        # skip pairs where either response produced no usable activations
        # (e.g. empty response string). Previously this would KeyError.
        if "abstain" not in activations or "non_abstain" not in activations:
            continue

        diff = activations["abstain"] - activations["non_abstain"]
        # move diff vectors to CPU to avoid accumulating thousands of
        # GPU tensors in the list, which can cause OOM on large datasets.
        diff_vectors_by_scenario[scenario].append(diff.cpu())

    # aggregate
    all_diffs = [d for diffs in diff_vectors_by_scenario.values() for d in diffs]

    if not all_diffs:
        print("No vectors extracted.")
        return

    if args.weighted:
        # scenario-weighted: compute per-scenario mean, then uniform average
        print("Using scenario-weighted aggregation:")
        scenario_means = []
        for scenario in sorted(diff_vectors_by_scenario.keys()):
            diffs = diff_vectors_by_scenario[scenario]
            scenario_mean = torch.stack(diffs).mean(dim=0)
            scenario_means.append(scenario_mean)
            marker = " (low count)" if len(diffs) < 30 else ""
            print(f"  {scenario:30s}  {len(diffs):5d} pairs{marker}")

        mean_steering_vector = torch.stack(scenario_means).mean(dim=0)
        print(f"\nAggregated {len(scenario_means)} scenario means into final vector.")
    else:
        # naive: global mean (original behavior)
        print("Using naive (global mean) aggregation.")
        stacked_diffs = torch.stack(all_diffs)
        mean_steering_vector = stacked_diffs.mean(dim=0)

    # optionally L2-normalize the final vector so that alpha controls perturbation
    # magnitude independent of layer-specific activation scale. Without this,
    # alpha=1.0 at layer 12 vs layer 23 are not comparable interventions,
    # and the layer sweep partly selects on vector magnitude rather than
    # direction quality
    raw_norm = mean_steering_vector.norm(p=2).item()
    print(f"Final vector L2 norm before normalization: {raw_norm:.6f}")

    if args.normalize:
        mean_steering_vector = mean_steering_vector / max(raw_norm, 1e-12)
        print("Applied L2 normalization to final steering vector.")

    # ensure final saved tensor is always float32 regardless of accumulation path
    mean_steering_vector = mean_steering_vector.float()

    # save
    out_dir = os.path.dirname(args.output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(mean_steering_vector, args.output_path)
    print(f"Saved steering vector for layer {args.layer_idx} to {args.output_path}")


def question_to_chat_ids(
    use_system_prompt: bool,
    question: str,
    tokenizer: AutoTokenizer,
    device: torch.device,
) -> torch.Tensor:
    """Tokenize the question with the chat template and generation prompt.

    Uses ``apply_chat_template(..., tokenize=True, add_generation_prompt=True)``
    so token boundaries match concatenation with separately tokenized responses.

    Args:
        use_system_prompt: If True, prepend ``SYSTEM_PROMPT`` as the system message.
        question: User message text.
        tokenizer: Hugging Face tokenizer.
        device: Device for returned token IDs.

    Returns:
        ``input_ids`` with shape ``(1, seq_len)`` on ``device``.
    """
    if use_system_prompt:
        prompt: List[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
    else:
        prompt = [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(
        prompt,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, required=True, help="HF model name or path"
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to training dataset file"
    )
    parser.add_argument(
        "--data_format",
        type=str,
        default="auto",
        choices=["auto", "csv", "json"],
        help="Dataset format. Use auto to infer from file extension.",
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="Path to save .pt vector"
    )
    parser.add_argument(
        "--layer_idx", type=int, required=True, help="Layer index to extract from"
    )
    parser.add_argument("--use_system_prompt", action="store_true")
    parser.add_argument(
        "--max_pairs", type=int, default=None, help="Max pairs to process"
    )
    parser.add_argument(
        "--weighted",
        action="store_true",
        help="Use scenario-weighted extraction (uniform weight per scenario)",
    )
    parser.add_argument(
        "--exclude_scenarios",
        type=str,
        default=None,
        help="Comma-separated scenarios to exclude, e.g. 'stale,subjective'",
    )
    parser.add_argument(
        "--response_tokens",
        type=int,
        default=10,
        help="Number of response tokens to average over",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="L2-normalize final steering vector before saving",
    )

    args = parser.parse_args()
    extract_vectors(args)
