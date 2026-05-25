"""
Convert tagged-responses JSON (new abstention detector) into flat benchmark CSVs.

Output schema matches data/grpo/final/csvs/*.csv and analysis.tables.AbstentionF1ScoreTable.
"""
from __future__ import annotations

import argparse
import json
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from analysis.load_results import _format_dataset_name, _scenario_label, post_training_stage
from recipe.abstention_datasets.abstract_abstention_dataset import Prompt
from recipe.abstention_datasets.coconot import CoCoNotDataset
from recipe.abstention_datasets.kuq import KUQDataset

EXCLUDED_DATASETS = (
    "KUQDataset_missing_category",
    "KUQDataset_counterfactual",
    "CoCoNotDataset_safety",
    "SelfAwareDataset",
    "NQDataset",
    "CoCoNotDataset_underspecification",
)

DEFAULT_MODEL_NAMES = {
    "tagged_responses_Gemma3_1B_vanilla.json": "gemma-3-1b-it-vanilla",
    "tagged_responses_Llama3.1_Tulu3.1_8B_vanilla.json": "llama-3.1-tulu-vanilla",
    "tagged_responses_Qwen2.5_0.5B_vanilla.json": "qwen-2.5-0.5b-vanilla",
    "tagged_responses_Qwen2.5_1.5B_vanilla.json": "qwen-2.5-1.5b-vanilla",
    "tagged_responses_Qwen2.5_3B_vanilla.json": "qwen-2.5-3b-vanilla",
    "tagged_responses_Qwen2.5_7B_vanilla.json": "qwen-2.5-7b-vanilla",
}


def _extend_kuq(dataset_name: str, prompt: Prompt) -> str:
    category = prompt.metadata.get("KUQ_category")
    if category is None:
        return f"{dataset_name}_missing_category"
    return f"{dataset_name}_{category.lower().replace(' ', '_')}"


def _extend_coconot(dataset_name: str, prompt: Prompt) -> str:
    category = prompt.metadata.get("CoCoNot_AbstentionBench_category")
    return f"{dataset_name}_{category.lower().replace(' ', '_')}"


@lru_cache(maxsize=2)
def _question_to_extended_dataset(dataset_name: str) -> dict[str, str]:
    if dataset_name == "KUQDataset":
        dataset_cls = KUQDataset
        extend_fn = _extend_kuq
    elif dataset_name == "CoCoNotDataset":
        dataset_cls = CoCoNotDataset
        extend_fn = _extend_coconot
    else:
        raise ValueError(f"No question map for dataset {dataset_name}")

    dataset = dataset_cls()
    return {
        dataset[i].question.strip(): extend_fn(dataset_name, dataset[i])
        for i in range(len(dataset))
    }


def resolve_dataset_name_extended(record: dict) -> str:
    dataset_name = record["dataset"]
    if dataset_name in {"KUQDataset", "CoCoNotDataset"}:
        question = record["question"].strip()
        extended = _question_to_extended_dataset(dataset_name).get(question)
        if extended is None:
            raise KeyError(
                f"Could not map {dataset_name} question to extended dataset name: "
                f"{question[:120]!r}"
            )
        return extended
    return dataset_name


def tagged_responses_to_dataframe(records: list[dict]) -> pd.DataFrame:
    rows = []
    for record in records:
        rows.append(
            {
                "dataset_name_extended": resolve_dataset_name_extended(record),
                "prompt_should_abstain": record["should_abstain"],
                "is_abstention": record["is_abstention"],
            }
        )

    df = pd.DataFrame(rows)
    for excluded in EXCLUDED_DATASETS:
        df = df[df["dataset_name_extended"] != excluded]
    return df


def aggregate_metrics(df: pd.DataFrame, model_name_formatted: str) -> pd.DataFrame:
    formatted = df.copy()
    formatted["model_name_formatted"] = model_name_formatted
    formatted["dataset_name_formatted"] = formatted["dataset_name_extended"].map(
        _format_dataset_name
    )
    formatted["scenario_label"] = formatted["dataset_name_extended"].map(_scenario_label)
    formatted["post_training_stage"] = post_training_stage(model_name_formatted)

    metrics = formatted.groupby(
        [
            "model_name_formatted",
            "scenario_label",
            "dataset_name_formatted",
            "post_training_stage",
        ],
        as_index=False,
    ).apply(
        lambda group: pd.Series(
            {
                "precision": precision_score(
                    group["prompt_should_abstain"],
                    group["is_abstention"],
                    zero_division=0.0,
                ),
                "recall": recall_score(
                    group["prompt_should_abstain"],
                    group["is_abstention"],
                    zero_division=0.0,
                ),
                "f1_score": f1_score(
                    group["prompt_should_abstain"],
                    group["is_abstention"],
                    zero_division=0.0,
                ),
            }
        ),
        include_groups=False,
    )
    return metrics.sort_values(
        ["scenario_label", "dataset_name_formatted"], kind="stable"
    ).reset_index(drop=True)


def infer_model_name(json_path: Path) -> str:
    if json_path.name in DEFAULT_MODEL_NAMES:
        return DEFAULT_MODEL_NAMES[json_path.name]

    stem = json_path.stem
    match = re.match(r"tagged_responses_(.+)_vanilla$", stem)
    if match:
        slug = match.group(1).lower().replace("_", "-")
        return f"{slug}-vanilla"
    return stem


def infer_output_path(json_path: Path, output: Path | None) -> Path:
    if output is not None:
        return output
    return json_path.with_name(json_path.stem.replace("tagged_responses_", "") + "_res.csv")


def convert_tagged_responses_json(
    json_path: Path,
    *,
    output_path: Path | None = None,
    model_name_formatted: str | None = None,
) -> pd.DataFrame:
    json_path = Path(json_path)
    with json_path.open() as handle:
        records = json.load(handle)

    model_name = model_name_formatted or infer_model_name(json_path)
    metrics = aggregate_metrics(tagged_responses_to_dataframe(records), model_name)

    out_path = infer_output_path(json_path, output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out_path, index=False)
    print(f"Wrote {len(metrics)} rows to {out_path}")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert tagged-responses JSON into flat benchmark CSV."
    )
    parser.add_argument("json_path", type=Path, help="Input tagged-responses JSON file.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: *_res.csv next to input).",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Value for model_name_formatted (default: inferred from filename).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_tagged_responses_json(
        args.json_path,
        output_path=args.output,
        model_name_formatted=args.model_name,
    )


if __name__ == "__main__":
    main()
