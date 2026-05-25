from pathlib import Path
from typing import Optional

import pandas as pd

from ..aggregate_metrics import aggregate_in_domain
from ..metrics_schema import BenchmarkRow
from ..rulebreakers_source import resolve_rulebreakers


def load_flat_csv_model(
    *,
    display_name: str,
    method: str,
    csv_path: Path,
    rulebreakers_csv: Optional[Path] = None,
) -> list[BenchmarkRow]:
    """Load ID and Rulebreakers metrics from a single-result flat CSV."""
    path = Path(csv_path)
    if not path.exists():
        print(f"Warning: Missing CSV for {display_name} ({method}): {path}")
        return []

    df = pd.read_csv(path)
    id_metrics = aggregate_in_domain(df)

    rows = [
        BenchmarkRow(
            model=display_name,
            method=method,
            split="in_domain",
            precision=id_metrics["precision"],
            recall=id_metrics["recall"],
            f1=id_metrics["f1_score"],
        )
    ]

    rb_metrics = resolve_rulebreakers(
        inline_df=df,
        rulebreakers_csv=rulebreakers_csv,
        model_label=f"{display_name} ({method})",
    )
    if rb_metrics is not None:
        rows.append(
            BenchmarkRow(
                model=display_name,
                method=method,
                split="rulebreakers",
                precision=rb_metrics["precision"],
                recall=rb_metrics["recall"],
                f1=rb_metrics["f1_score"],
            )
        )

    return rows


def load_flat_csv_method(
    models: dict,
    method: str,
) -> list[BenchmarkRow]:
    """Load all models for a flat-CSV method from config dict."""
    rows: list[BenchmarkRow] = []
    for entry in models.values():
        rows.extend(
            load_flat_csv_model(
                display_name=entry["display_name"],
                method=method,
                csv_path=entry["csv"],
                rulebreakers_csv=entry.get("rulebreakers_csv"),
            )
        )
    return rows
