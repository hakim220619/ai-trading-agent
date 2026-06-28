"""Standalone M1 nearest supply/demand scalping strategy for XAUUSD."""
from __future__ import annotations

import pandas as pd

from app.ml.feature_engineering import build_features
from app.strategy.scalping_common import ScalpingPlan, ScalpingSignal, build_scalping_plan


def prepare_snd_m1_features(df: pd.DataFrame, zone_period: int = 30) -> pd.DataFrame:
    """Build prior-bar S/D zones and confirmation features without look-ahead."""
    out = build_features(df)
    out["ema9"] = out["close"].ewm(span=9, adjust=False).mean()
    out["ema21"] = out["close"].ewm(span=21, adjust=False).mean()
    out["demand_low"] = out["low"].shift(1).rolling(zone_period).min()
    out["supply_high"] = out["high"].shift(1).rolling(zone_period).max()
    out["demand_high"] = out["demand_low"] + out["atr"] * 0.35
    out["supply_low"] = out["supply_high"] - out["atr"] * 0.35
    out["prior_demand_low"] = out["demand_low"].shift(1)
    out["prior_supply_high"] = out["supply_high"].shift(1)
    out["previous_close"] = out["close"].shift(1)
    out["previous_rsi"] = out["rsi"].shift(1)
    out["previous_macd_hist"] = out["macd_hist"].shift(1)
    out["volume_ratio"] = out["volume"] / out["volume_avg"].replace(0, pd.NA)
    out["atr_ratio"] = out["atr"] / out["close"].replace(0, pd.NA)
    return out


def generate_snd_m1_signal(
    row: pd.Series,
    symbol: str,
    spread_points: float,
    max_spread_points: float,
) -> ScalpingSignal:
    """Enter at the nearest zone after continuation or reversal confirmation."""
    signal = ScalpingSignal(
        price=float(row.get("close", 0) or 0),
        atr=float(row.get("atr", 0) or 0),
    )
    if "XAU" not in symbol.upper() and "GOLD" not in symbol.upper():
        signal.reasons = ["SND M1 strategy is restricted to XAU/GOLD"]
        return signal
    required = (
        "ema9", "ema21", "ema50", "ema200", "rsi", "previous_rsi",
        "macd_hist", "previous_macd_hist", "demand_low", "demand_high",
        "supply_low", "supply_high", "volume_ratio", "atr_ratio",
        "prior_demand_low", "prior_supply_high",
        "open", "high", "low", "close", "upper_wick", "lower_wick",
    )
    if any(pd.isna(row.get(name)) for name in required) or signal.atr <= 0:
        signal.reasons = ["indicators or zones not ready"]
        return signal
    if spread_points > max_spread_points:
        signal.reasons = [f"spread too high ({spread_points:.1f})"]
        return signal

    close, open_price = float(row["close"]), float(row["open"])
    low, high = float(row["low"]), float(row["high"])
    ema9, ema21, ema50, ema200 = (float(row[x]) for x in ("ema9", "ema21", "ema50", "ema200"))
    demand_low, demand_high = float(row["demand_low"]), float(row["demand_high"])
    supply_low, supply_high = float(row["supply_low"]), float(row["supply_high"])
    prior_demand, prior_supply = float(row["prior_demand_low"]), float(row["prior_supply_high"])
    rsi, previous_rsi = float(row["rsi"]), float(row["previous_rsi"])
    macd, previous_macd = float(row["macd_hist"]), float(row["previous_macd_hist"])
    body = max(float(row.get("candle_body", 0) or 0), signal.atr * 0.05)
    demand_test = low <= demand_high and close > demand_low
    supply_test = high >= supply_low and close < supply_high
    bullish_rejection = close > open_price and float(row["lower_wick"]) >= body * 0.8
    bearish_rejection = close < open_price and float(row["upper_wick"]) >= body * 0.8
    candle_range = max(high - low, 1e-12)
    close_location = (close - low) / candle_range
    trend_up = ema9 > ema21 > ema50 > ema200
    trend_down = ema9 < ema21 < ema50 < ema200
    flipped_supply_retest = float(row["previous_close"]) > prior_supply and low <= prior_supply + signal.atr * 0.20 and close > prior_supply
    flipped_demand_retest = float(row["previous_close"]) < prior_demand and high >= prior_demand - signal.atr * 0.20 and close < prior_demand
    activity_ok = float(row["volume_ratio"]) >= 1.10 and 0.00003 <= float(row["atr_ratio"]) <= 0.02

    continuation_buy = flipped_supply_retest and trend_up and bullish_rejection and close_location >= 0.65 and 50 <= rsi <= 68 and macd > 0
    continuation_sell = flipped_demand_retest and trend_down and bearish_rejection and close_location <= 0.35 and 32 <= rsi <= 50 and macd < 0
    reversal_buy = (
        demand_test and bullish_rejection and close_location >= 0.75
        and close > ema200 and previous_rsi <= 45 < rsi
        and previous_macd <= 0 < macd and close > ema9 > ema21
    )
    reversal_sell = (
        supply_test and bearish_rejection and close_location <= 0.25
        and close < ema200 and previous_rsi >= 55 > rsi
        and previous_macd >= 0 > macd and close < ema9 < ema21
    )
    if activity_ok and (continuation_buy or reversal_buy):
        setup = "trend continuation" if continuation_buy else "demand reversal"
        signal.action = "BUY"
        signal.reasons = [setup, f"demand {demand_low:.3f}-{demand_high:.3f}", "volume/ATR confirmed"]
    elif activity_ok and (continuation_sell or reversal_sell):
        setup = "trend continuation" if continuation_sell else "supply reversal"
        signal.action = "SELL"
        signal.reasons = [setup, f"supply {supply_low:.3f}-{supply_high:.3f}", "volume/ATR confirmed"]
    else:
        signal.reasons = [
            f"demand test={demand_test}, buy confirm={continuation_buy or reversal_buy}",
            f"supply test={supply_test}, sell confirm={continuation_sell or reversal_sell}",
        ]
    return signal


def build_snd_m1_plan(
    direction: str,
    entry: float,
    atr_value: float,
    balance: float,
    symbol: str,
    risk_percent: float = 0.5,
) -> ScalpingPlan:
    """Use a zone-tolerant 1.4 ATR stop and 1.6R target."""
    return build_scalping_plan(direction, entry, atr_value, balance, symbol, risk_percent, 1.40, 1.60)
