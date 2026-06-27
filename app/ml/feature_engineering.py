"""Feature engineering for the XGBoost model."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.strategy.indicators import add_indicators

# The exact, ordered feature set used for both training and inference.
FEATURE_COLUMNS: list[str] = [
    "return_1",
    "return_3",
    "return_5",
    "ema20",
    "ema50",
    "ema200",
    "rsi",
    "macd",
    "atr",
    "bb_upper",
    "bb_lower",
    "candle_body",
    "upper_wick",
    "lower_wick",
    "volume",
    "spread",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add indicators + return features to ``df``.

    Returns a copy containing every FEATURE_COLUMN (plus original OHLCV).
    """
    out = add_indicators(df)
    close = out["close"]

    out["return_1"] = close.pct_change(1)
    out["return_3"] = close.pct_change(3)
    out["return_5"] = close.pct_change(5)

    if "spread" not in out.columns:
        out["spread"] = 0.0
    if "volume" not in out.columns:
        out["volume"] = 0.0

    return out


def build_target(df: pd.DataFrame, horizon: int = 1, atr_mult: float = 0.5) -> pd.Series:
    """Binary target: 1 if next candle rises more than ATR*atr_mult, else 0.

    Compares close[t+horizon] vs close[t]. The last ``horizon`` rows get 0
    (no future data) and should be dropped before training.
    """
    close = df["close"]
    atr_v = df["atr"] if "atr" in df.columns else (df["high"] - df["low"])
    future = close.shift(-horizon)
    move = future - close
    target = (move > (atr_v * atr_mult)).astype(int)
    return target


def make_dataset(df: pd.DataFrame, horizon: int = 1, atr_mult: float = 0.5) -> tuple[pd.DataFrame, pd.Series]:
    """Produce a clean (X, y) pair ready for training.

    Drops warm-up NaNs and the tail rows that have no future target.
    """
    feat = build_features(df)
    feat["target"] = build_target(feat, horizon=horizon, atr_mult=atr_mult)

    feat = feat.replace([np.inf, -np.inf], np.nan)
    feat = feat.dropna(subset=FEATURE_COLUMNS + ["target"])
    if horizon > 0:
        feat = feat.iloc[:-horizon] if len(feat) > horizon else feat

    X = feat[FEATURE_COLUMNS].copy()
    y = feat["target"].astype(int).copy()
    return X, y


def features_to_row(features: dict[str, float]) -> pd.DataFrame:
    """Build a single-row inference DataFrame from a feature dict.

    Missing features default to 0.0 so inference never crashes.
    """
    row = {col: float(features.get(col, 0.0)) for col in FEATURE_COLUMNS}
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)
