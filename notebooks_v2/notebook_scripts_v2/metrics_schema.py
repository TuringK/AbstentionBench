from dataclasses import dataclass
from typing import Literal, Optional

import pandas as pd

Split = Literal["in_domain", "rulebreakers"]


@dataclass
class BenchmarkRow:
    model: str
    method: str
    split: Split
    precision: float
    recall: float
    f1: float
    best_coeff: Optional[float] = None
    best_layer: Optional[int] = None


def to_plot_dataframe(rows: list[BenchmarkRow]) -> pd.DataFrame:
    """Convert rows to the bar-chart schema used by v1 plots."""
    records = []
    for row in rows:
        record = {
            "Model": row.model,
            "Method": row.method,
            "Split": row.split,
            "Precision": row.precision,
            "Recall": row.recall,
            "F1": row.f1,
        }
        if row.best_coeff is not None:
            record["Best Coeff"] = row.best_coeff
        if row.best_layer is not None:
            record["Best Layer"] = row.best_layer
        records.append(record)
    return pd.DataFrame(records)


def best_combinations_dataframe(rows: list[BenchmarkRow]) -> pd.DataFrame:
    """CAA best-config table (ID rows with coeff/layer metadata)."""
    records = []
    for row in rows:
        if row.split != "in_domain" or row.best_coeff is None:
            continue
        records.append(
            {
                "Model": row.model,
                "Best Coeff": row.best_coeff,
                "Best Layer": row.best_layer,
                "Precision": row.precision,
                "Recall": row.recall,
                "F1": row.f1,
            }
        )
    return pd.DataFrame(records)
