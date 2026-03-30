from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def safe_scalar(value: Any, default: float = 0.0) -> float:
    """Convert any value to a scalar float safely."""
    if value is None:
        return default
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if isinstance(value, pd.Series):
        if len(value) == 0:
            return default
        return float(value.iloc[-1])
    if isinstance(value, pd.DataFrame):
        if value.empty:
            return default
        return float(value.iloc[-1, 0]) if value.shape[1] > 0 else default
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return default
        return float(value.flat[-1])
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_get_series(df: pd.DataFrame, col: str, default: float = 0.0) -> float:
    """Safely get a scalar value from a DataFrame column."""
    if df is None or df.empty:
        return default
    if col not in df.columns:
        return default
    return safe_scalar(df[col], default)


def safe_compare(val1: Any, op: str, val2: Any) -> bool:
    """Safely compare two values, handling pandas Series."""
    v1 = safe_scalar(val1)
    v2 = safe_scalar(val2)
    if op == ">":
        return v1 > v2
    if op == "<":
        return v1 < v2
    if op == ">=":
        return v1 >= v2
    if op == "<=":
        return v1 <= v2
    if op == "==":
        return v1 == v2
    if op == "!=":
        return v1 != v2
    return False
