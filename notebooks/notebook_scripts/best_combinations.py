import pandas as pd
from pathlib import Path
from typing import Union, Optional

from .summarize_metrics import summarize_layer_metrics


def best_combinations(
    models: dict,
    coeffs: list,
    results_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    save: bool = False,
    print_table: bool = True,
) -> pd.DataFrame:
    """Find the best vector combinations for each model. Optionally save to CSV."""
    results_dir = Path(results_dir)
    results = []

    for model_key, model_name in models.items():
        best_f1 = -1
        best_row = None
        best_coeff = None

        for coeff in coeffs:
            if coeff == 1.0:
                csv_path = results_dir / f"{model_key}_v1.csv"
            else:
                # 2.0 -> 2_0
                coeff_str = f"{coeff:.1f}".replace(".", "_")
                csv_path = (
                    results_dir
                    / f"{model_key}_sweep"
                    / f"{model_key}_v1_sweep_{coeff_str}.csv"
                )

            if not csv_path.exists():
                print(f"Warning: Missing file: {csv_path}")
                continue

            try:
                summary = summarize_layer_metrics(
                    str(csv_path), model_name=model_name, filter_incomplete=True
                )
                if summary.empty:
                    print(f"Warning: Empty summary for {csv_path}")
                    continue

                # summarize_layer_metrics sorts by 'Avg F1 Score' descending
                top_row = summary.iloc[0]
                if top_row["Avg F1 Score"] > best_f1:
                    best_f1 = top_row["Avg F1 Score"]
                    best_row = top_row
                    best_coeff = coeff
            except Exception as e:
                print(f"Error processing {csv_path}: {e}")

        if best_row is not None:
            results.append(
                {
                    "Model": model_name,
                    "Best Coeff": best_coeff,
                    "Best Layer": int(best_row[f"{model_name} layers"]),
                    "Precision": best_row["Avg Precision"],
                    "Recall": best_row["Avg Recall"],
                    "F1": best_row["Avg F1 Score"],
                }
            )

    if not results:
        print("No valid data found to print best combinations.")
        return

    final_df = pd.DataFrame(results)

    if print_table:
        print("Final Best Combinations per Model:")
        print("-" * 90)
        print(final_df.to_markdown(index=False))
        print("-" * 90)

    if save and output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "best_combinations.csv"
        final_df.to_csv(out_path, index=False)
        print(f"\nSaved summary to {out_path}")

    return final_df
