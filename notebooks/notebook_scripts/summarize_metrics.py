import pandas as pd
from pathlib import Path
from typing import Union, Optional

REQUIRED_DATASETS = [
    "ALCUNA",
    "BB/Disambiguate",
    "BB/Known unknowns",
    "BBQ",
    "CoCoNot/False presumptions",
    "CoCoNot/Humanizing",
    "CoCoNot/Incomprehensible",
    "CoCoNot/Subjective",
    "CoCoNot/Temporal",
    "CoCoNot/Unknowns",
    "CoCoNot/Unsupported",
    "FalseQA",
    "FreshQA",
    "GPQA-Diamond",
    "GSM8K",
    "KUQ/Ambiguous",
    "KUQ/Controversial",
    "KUQ/False assumptions",
    "KUQ/Future unknowns",
    "KUQ/Unsolved problems",
    "MMLU History",
    "MMLU Math",
    "MediQ",
    "MoralChoice",
    "Musique",
    "QASPER",
    "QAQA",
    "SQuAD 2.0",
    "SituatedQA/Geo",
    "UMWP",
    "WorldSense",
]


def steering_results_csv_path(
    results_dir: Union[str, Path],
    model_key: str,
    coeff: float,
    *,
    csv_version: str = "v3",
    baseline_csv_at_root: bool = False,
) -> Path:
    """Resolve path to a steering sweep CSV.

    `baseline_csv_at_root=True` matches older layouts (e.g. v2): coefficient 1.0
    lives at `{results_dir}/{model_key}_{csv_version}.csv`. Other coeffs live under
    `{model_key}_sweep/`. With `baseline_csv_at_root=False` (v3), every coeff
    including 1.0 lives at `{model_key}_sweep/{model_key}_{csv_version}_sweep_{coeff}.csv`.
    """
    results_dir = Path(results_dir)
    coeff_str = f"{coeff:.1f}".replace(".", "_")
    if baseline_csv_at_root and coeff == 1.0:
        return results_dir / f"{model_key}_{csv_version}.csv"
    return (
        results_dir
        / f"{model_key}_sweep"
        / f"{model_key}_{csv_version}_sweep_{coeff_str}.csv"
    )


def filter_incomplete_layers(df: pd.DataFrame) -> pd.DataFrame:
    required = set(REQUIRED_DATASETS)
    counts = (
        df[df["dataset_name_formatted"].isin(required)]
        .groupby("vector_index")["dataset_name_formatted"]
        .nunique()
    )
    complete_layers = counts[counts == len(required)].index

    # all_layers = set(df["vector_index"].unique())
    # filtered_layers = all_layers - set(complete_layers)
    # if filtered_layers:
    #     print(
    #         f"Filtered out layers missing required datasets: {sorted(list(filtered_layers))}"
    #     )

    return df[df["vector_index"].isin(complete_layers)]


def summarize_layer_metrics(
    csv_path: str,
    model_name: str = "Qwen 2.5 0.5B Instruct",
    filter_incomplete: bool = False,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if filter_incomplete:
        df = filter_incomplete_layers(df)

    summary = (
        df.groupby("vector_index")[["precision", "recall", "f1_score"]]
        .mean()
        .reset_index()
    )

    summary = summary.rename(
        columns={
            "vector_index": f"{model_name} layers",
            "precision": "Avg Precision",
            "recall": "Avg Recall",
            "f1_score": "Avg F1 Score",
        }
    )

    summary["Avg Precision"] = summary["Avg Precision"].round(4)
    summary["Avg Recall"] = summary["Avg Recall"].round(4)
    summary["Avg F1 Score"] = summary["Avg F1 Score"].round(4)

    summary = summary.sort_values(by="Avg F1 Score", ascending=False)

    return summary


def print_model_vectors(
    model_key: str,
    model_name: str,
    coeff: float,
    results_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    save: bool = False,
    *,
    csv_version: str = "v3",
    baseline_csv_at_root: bool = False,
) -> None:
    """Read a specific model/coefficient vectors result and print it. Optionally save."""
    csv_path = steering_results_csv_path(
        results_dir,
        model_key,
        coeff,
        csv_version=csv_version,
        baseline_csv_at_root=baseline_csv_at_root,
    )

    if not csv_path.exists():
        print(f"Warning: Missing file: {csv_path}")
        return

    try:
        summary = summarize_layer_metrics(
            str(csv_path), model_name=model_name, filter_incomplete=True
        )
        if summary.empty:
            print(f"Warning: Empty summary for {csv_path}")
            return

        print(f"Metrics for {model_name} (Coefficient: {coeff})")
        print("-" * 60)
        print(summary.to_markdown(index=False))
        print("-" * 60)

        if save and output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            out_path = output_dir / f"{model_key}_coeff_{coeff}_summary.csv"
            summary.to_csv(out_path, index=False)
            print(f"\nSaved summary to {out_path}")

    except Exception as e:
        print(f"Error processing {csv_path}: {e}")
