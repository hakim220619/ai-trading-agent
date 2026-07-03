"""Standalone RSI reversal strategy for intraday backtesting."""
from __future__ import annotations

import pandas as pd

from app.ml.feature_engineering import build_features
from app.strategy.scalping_common import ScalpingPlan, ScalpingSignal, build_scalping_plan


def prepare_rsi_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add indicators and previous RSI needed for fresh threshold crosses."""
    out = build_features(df)
    out["previous_rsi"] = out["rsi"].shift(1)
    return out


def generate_rsi_signal(
    row: pd.Series,
    symbol: str,
    spread_points: float,
    max_spread_points: float,
    oversold: float = 30.0,
    overbought: float = 70.0,
) -> ScalpingSignal:
    """Generate BUY/SELL on RSI exits from oversold/overbought zones."""
    signal = ScalpingSignal(
        price=float(row.get("close", 0) or 0),
        atr=float(row.get("atr", 0) or 0),
    )
    required = ("rsi", "previous_rsi", "atr")
    if any(pd.isna(row.get(name)) for name in required) or signal.atr <= 0:
        signal.reasons = ["RSI strategy indicators not ready"]
        return signal
    if spread_points > max_spread_points:
        signal.reasons = [f"spread too high ({spread_points:.1f})"]
        return signal

    rsi = float(row["rsi"])
    previous_rsi = float(row["previous_rsi"])
    if previous_rsi <= oversold < rsi:
        signal.action = "BUY"
        signal.reasons = [f"RSI rebound from oversold ({previous_rsi:.1f}->{rsi:.1f})"]
    elif previous_rsi >= overbought > rsi:
        signal.action = "SELL"
        signal.reasons = [f"RSI rejection from overbought ({previous_rsi:.1f}->{rsi:.1f})"]
    else:
        signal.reasons = [f"no fresh RSI cross ({rsi:.1f})"]
    return signal


def build_rsi_plan(
    direction: str,
    entry: float,
    atr_value: float,
    balance: float,
    symbol: str,
    risk_percent: float = 0.5,
) -> ScalpingPlan:
    return build_scalping_plan(direction, entry, atr_value, balance, symbol, risk_percent, 1.20, 1.40)
