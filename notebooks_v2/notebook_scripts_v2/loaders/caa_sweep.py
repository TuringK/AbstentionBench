from pathlib import Path
from typing import Optional

import pandas as pd

from ..aggregate_metrics import aggregate_in_domain, summarize_layers_id_only
from ..metrics_schema import BenchmarkRow
from ..rulebreakers_source import resolve_rulebreakers


def find_best_caa_config(
    sweep_csvs: dict[float, Path],
) -> Optional[tuple[float, int, dict[str, float], pd.DataFrame]]:
    """Find best (coeff, layer) by ID F1 only. Returns coeff, layer, metrics, best-coeff df."""
    best_f1 = -1.0
    best_coeff: Optional[float] = None
    best_layer: Optional[int] = None
    best_metrics: Optional[dict[str, float]] = None
    best_coeff_df: Optional[pd.DataFrame] = None

    for coeff, csv_path in sweep_csvs.items():
        path = Path(csv_path)
        if not path.exists():
            print(f"Warning: Missing sweep CSV: {path}")
            continue

        df = pd.read_csv(path)
        layer_summary = summarize_layers_id_only(df)
        if layer_summary.empty:
            print(f"Warning: No complete layers in {path}")
            continue

        top = layer_summary.loc[layer_summary["f1_score"].idxmax()]
        if top["f1_score"] > best_f1:
            best_f1 = float(top["f1_score"])
            best_coeff = coeff
            best_layer = int(top["vector_index"])
            best_metrics = {
                "precision": float(top["precision"]),
                "recall": float(top["recall"]),
                "f1_score": float(top["f1_score"]),
            }
            best_coeff_df = df

    if best_coeff is None or best_layer is None or best_metrics is None:
        return None

    return best_coeff, best_layer, best_metrics, best_coeff_df


def load_caa_model(
    *,
    display_name: str,
    sweep_csvs: dict[float, Path],
    rulebreakers_csv: Optional[Path] = None,
) -> list[BenchmarkRow]:
    """Two-phase CAA load: ID-only best config selection, then Rulebreakers lookup."""
    result = find_best_caa_config(sweep_csvs)
    if result is None:
        print(f"Warning: No valid CAA sweep data for {display_name}")
        return []

    best_coeff, best_layer, id_metrics, best_coeff_df = result

    rows = [
        BenchmarkRow(
            model=display_name,
            method="CAA",
            split="in_domain",
            precision=id_metrics["precision"],
            recall=id_metrics["recall"],
            f1=id_metrics["f1_score"],
            best_coeff=best_coeff,
            best_layer=best_layer,
        )
    ]

    rb_metrics = resolve_rulebreakers(
        inline_df=best_coeff_df,
        rulebreakers_csv=rulebreakers_csv,
        vector_index=best_layer,
        model_label=f"{display_name} (CAA)",
    )
    if rb_metrics is not None:
        rows.append(
            BenchmarkRow(
                model=display_name,
                method="CAA",
                split="rulebreakers",
                precision=rb_metrics["precision"],
                recall=rb_metrics["recall"],
                f1=rb_metrics["f1_score"],
                best_coeff=best_coeff,
                best_layer=best_layer,
            )
        )

    return rows


def load_caa_sweep_data(
    models: dict,
) -> tuple[list[BenchmarkRow], pd.DataFrame]:
    """Load all CAA models; return benchmark rows and sweep summary for diagnostics."""
    rows: list[BenchmarkRow] = []
    sweep_frames: list[pd.DataFrame] = []

    for entry in models.values():
        display_name = entry["display_name"]
        sweep_csvs = entry["sweep_csvs"]
        for coeff, csv_path in sweep_csvs.items():
            path = Path(csv_path)
            if not path.exists():
                continue
            df = pd.read_csv(path)
            layer_summary = summarize_layers_id_only(df)
            if layer_summary.empty:
                continue
            layer_summary["model"] = display_name
            layer_summary["coeff"] = coeff
            sweep_frames.append(layer_summary)

        rows.extend(
            load_caa_model(
                display_name=display_name,
                sweep_csvs=sweep_csvs,
                rulebreakers_csv=entry.get("rulebreakers_csv"),
            )
        )

    df_all = pd.concat(sweep_frames, ignore_index=True) if sweep_frames else pd.DataFrame()
    return rows, df_all


def load_caa_method(models: dict) -> list[BenchmarkRow]:
    rows, _ = load_caa_sweep_data(models)
    return rows
