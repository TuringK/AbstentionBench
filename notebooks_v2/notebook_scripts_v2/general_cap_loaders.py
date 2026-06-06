from __future__ import annotations

from typing import Optional

import pandas as pd


def resolve_method_value(record: pd.Series, method: str) -> Optional[float]:
    """Return a method value from a CSV row, accepting legacy ``Steering`` for ``CAA``."""
    if method in record.index and pd.notna(record[method]):
        return float(record[method])
    if method == "CAA" and "Steering" in record.index and pd.notna(record["Steering"]):
        return float(record["Steering"])
    return None
