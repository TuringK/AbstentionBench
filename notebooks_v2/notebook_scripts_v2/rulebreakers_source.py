from pathlib import Path
from typing import Optional

import pandas as pd

from .aggregate_metrics import extract_rulebreakers


def resolve_rulebreakers(
    *,
    inline_df: Optional[pd.DataFrame] = None,
    rulebreakers_csv: Optional[Path] = None,
    vector_index: Optional[int] = None,
    model_label: str = "",
) -> Optional[dict[str, float]]:
    """Resolve Rulebreakers metrics: inline row first, then optional fallback CSV."""
    if inline_df is not None and not inline_df.empty:
        metrics = extract_rulebreakers(inline_df, vector_index=vector_index)
        if metrics is not None:
            return metrics

    if rulebreakers_csv is not None:
        path = Path(rulebreakers_csv)
        if path.exists():
            df = pd.read_csv(path)
            metrics = extract_rulebreakers(df, vector_index=vector_index)
            if metrics is not None:
                return metrics
            if not df.empty and "f1_score" in df.columns:
                row = df.iloc[0]
                return {
                    "precision": float(row["precision"]),
                    "recall": float(row["recall"]),
                    "f1_score": float(row["f1_score"]),
                }
        else:
            print(f"Warning: Rulebreakers CSV not found for {model_label}: {path}")

    print(f"Warning: No Rulebreakers metrics for {model_label}")
    return None
