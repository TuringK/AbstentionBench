from typing import Optional

import pandas as pd

from .benchmark_datasets import REQUIRED_DATASETS, RULEBREAKERS_DATASET


def id_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Rows belonging to the in-domain benchmark (excludes Rulebreakers)."""
    return df[df["dataset_name_formatted"].isin(REQUIRED_DATASETS)]


def filter_incomplete_layers(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only layers that have all required ID datasets."""
    if "vector_index" not in df.columns:
        return df

    required = set(REQUIRED_DATASETS)
    counts = (
        id_rows(df)
        .groupby("vector_index")["dataset_name_formatted"]
        .nunique()
    )
    complete_layers = counts[counts == len(required)].index
    return df[df["vector_index"].isin(complete_layers)]


def aggregate_in_domain(df: pd.DataFrame) -> dict[str, float]:
    """Macro-average precision, recall, f1 over required ID datasets."""
    subset = id_rows(df)
    if subset.empty:
        return {"precision": float("nan"), "recall": float("nan"), "f1_score": float("nan")}

    return {
        "precision": float(subset["precision"].mean()),
        "recall": float(subset["recall"].mean()),
        "f1_score": float(subset["f1_score"].mean()),
    }


def extract_rulebreakers(
    df: pd.DataFrame,
    *,
    vector_index: Optional[int] = None,
) -> Optional[dict[str, float]]:
    """Return Rulebreakers metrics from inline rows, optionally at a specific layer."""
    rb = df[df["dataset_name_formatted"] == RULEBREAKERS_DATASET]
    if rb.empty:
        return None

    if vector_index is not None and "vector_index" in rb.columns:
        rb = rb[rb["vector_index"] == vector_index]
        if rb.empty:
            return None

    row = rb.iloc[0]
    return {
        "precision": float(row["precision"]),
        "recall": float(row["recall"]),
        "f1_score": float(row["f1_score"]),
    }


def summarize_layers_id_only(df: pd.DataFrame) -> pd.DataFrame:
    """Per-layer ID macro averages (Rulebreakers excluded)."""
    filtered = filter_incomplete_layers(df)
    id_df = id_rows(filtered)
    if id_df.empty:
        return pd.DataFrame()

    return (
        id_df.groupby("vector_index")[["precision", "recall", "f1_score"]]
        .mean()
        .reset_index()
    )
