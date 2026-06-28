"""Pure EMA9/EMA21 crossover strategy for M1 scalping."""
from __future__ import annotations

import pandas as pd

from app.ml.feature_engineering import build_features
from app.strategy.scalping_common import ScalpingPlan, ScalpingSignal, build_scalping_plan


def prepare_ma_m1_features(df: pd.DataFrame) -> pd.DataFrame:
    out = build_features(df)
    out["ema9"] = out["close"].ewm(span=9, adjust=False).mean()
    out["ema21"] = out["close"].ewm(span=21, adjust=False).mean()
    out["previous_ema9"] = out["ema9"].shift(1)
    out["previous_ema21"] = out["ema21"].shift(1)
    return out


def generate_ma_m1_signal(
    row: pd.Series,
    symbol: str,
    spread_points: float,
    max_spread_points: float,
) -> ScalpingSignal:
    signal = ScalpingSignal(
        price=float(row.get("close", 0) or 0),
        atr=float(row.get("atr", 0) or 0),
    )
    required = ("ema9", "ema21", "previous_ema9", "previous_ema21", "atr")
    if any(pd.isna(row.get(name)) for name in required) or signal.atr <= 0:
        signal.reasons = ["moving averages not ready"]
        return signal
    if spread_points > max_spread_points:
        signal.reasons = [f"spread too high ({spread_points:.1f})"]
        return signal
    ema9, ema21 = float(row["ema9"]), float(row["ema21"])
    previous_ema9 = float(row["previous_ema9"])
    previous_ema21 = float(row["previous_ema21"])
    if previous_ema9 <= previous_ema21 and ema9 > ema21:
        signal.action = "BUY"
        signal.reasons = ["EMA9 crossed above EMA21"]
    elif previous_ema9 >= previous_ema21 and ema9 < ema21:
        signal.action = "SELL"
        signal.reasons = ["EMA9 crossed below EMA21"]
    else:
        signal.reasons = ["no fresh EMA9/EMA21 crossover"]
    return signal


def build_ma_m1_plan(
    direction: str,
    entry: float,
    atr_value: float,
    balance: float,
    symbol: str,
    risk_percent: float = 0.5,
) -> ScalpingPlan:
    return build_scalping_plan(direction, entry, atr_value, balance, symbol, risk_percent, 1.10, 1.30)
