"""Generic helper utilities used across the project."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd


def utc_now() -> datetime:
    """Timezone-aware current UTC time."""
    return datetime.now(tz=timezone.utc)


def round_to_step(value: float, step: float) -> float:
    """Round ``value`` down to the nearest multiple of ``step``.

    Used for lot sizes that must respect the broker's volume step.
    """
    if step <= 0:
        return value
    return float(np.floor(value / step) * step)


def clamp(value: float, low: float, high: float) -> float:
    """Constrain ``value`` to the inclusive range [low, high]."""
    return max(low, min(high, value))


def safe_float(value: Any, default: float = 0.0) -> float:
    """Convert ``value`` to float, returning ``default`` on failure."""
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def pct_change(old: float, new: float) -> float:
    """Percentage change from ``old`` to ``new`` (0 if old is 0)."""
    if old == 0:
        return 0.0
    return (new - old) / old * 100.0


def dataframe_to_records(df: pd.DataFrame, limit: int | None = None) -> list[dict[str, Any]]:
    """Convert a DataFrame to JSON-serialisable records (timestamps -> isoformat)."""
    if df is None or df.empty:
        return []
    out = df.tail(limit) if limit else df
    out = out.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].astype(str)
    return out.replace({np.nan: None}).to_dict(orient="records")


def describe_decision(signal: str, reasons: list[str]) -> str:
    """Human-readable one-line summary of a trading decision."""
    joined = "; ".join(reasons) if reasons else "no reasons recorded"
    return f"[{signal.upper()}] {joined}"
