"""Standalone M5 trend-breakout scalping strategy for all liquid markets."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from app.ml.feature_engineering import build_features
from app.strategy.scalping_common import (
    ScalpingPlan,
    ScalpingSignal,
    build_scalping_plan,
    market_family,
)


@dataclass(frozen=True)
class M5Preset:
    breakout_period: int
    stop_atr: float
    risk_reward: float
    min_volume_ratio: float
    max_atr_ratio: float
    min_trend_strength: float
    min_body_atr: float


def get_m5_preset(symbol: str) -> M5Preset:
    family = market_family(symbol)
    if family == "crypto":
        return M5Preset(12, 1.60, 1.70, 1.10, 0.04, 0.16, 0.25)
    if family == "metals":
        return M5Preset(10, 1.35, 1.60, 1.05, 0.02, 0.18, 0.30)
    return M5Preset(12, 1.25, 1.50, 1.05, 0.012, 0.16, 0.25)


def prepare_m5_features(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Add M5 trend and prior-bar breakout levels without look-ahead bias."""
    out = build_features(df)
    period = get_m5_preset(symbol).breakout_period
    out["ema9"] = out["close"].ewm(span=9, adjust=False).mean()
    out["ema21"] = out["close"].ewm(span=21, adjust=False).mean()
    out["breakout_high"] = out["high"].shift(1).rolling(period).max()
    out["breakout_low"] = out["low"].shift(1).rolling(period).min()
    out["atr_ratio"] = out["atr"] / out["close"].replace(0, pd.NA)
    out["volume_ratio"] = out["volume"] / out["volume_avg"].replace(0, pd.NA)
    out["previous_close"] = out["close"].shift(1)
    out["previous_ema9"] = out["ema9"].shift(1)
    out["previous_ema21"] = out["ema21"].shift(1)
    out["ema21_slope"] = out["ema21"].diff()
    out["trend_strength"] = (out["ema9"] - out["ema21"]).abs() / out["atr"].replace(0, pd.NA)
    candle_range = (out["high"] - out["low"]).replace(0, pd.NA)
    out["close_location"] = (out["close"] - out["low"]) / candle_range
    return out


def generate_m5_signal(row: pd.Series, symbol: str, spread_points: float, max_spread_points: float) -> ScalpingSignal:
    """Trade M5 setups; the full ensemble is reserved for XAU/GOLD symbols."""
    signal = ScalpingSignal(price=float(row.get("close", 0) or 0), atr=float(row.get("atr", 0) or 0))
    required = (
        "ema9", "ema21", "ema50", "ema200", "rsi", "macd_hist",
        "breakout_high", "breakout_low", "atr_ratio", "volume_ratio",
        "previous_close", "previous_ema9", "previous_ema21", "ema21_slope",
        "trend_strength", "close_location",
    )
    if any(pd.isna(row.get(name)) for name in required) or signal.atr <= 0:
        signal.reasons = ["indicators not ready"]
        return signal
    if spread_points > max_spread_points:
        signal.reasons = [f"spread too high ({spread_points:.1f})"]
        return signal

    p = get_m5_preset(symbol)
    close, ema9, ema21, ema50, ema200 = (float(row[x]) for x in ("close", "ema9", "ema21", "ema50", "ema200"))
    rsi, macd, atr_ratio = (float(row[x]) for x in ("rsi", "macd_hist", "atr_ratio"))
    body_atr = float(row.get("candle_body", 0) or 0) / signal.atr
    breakout_high, breakout_low = float(row["breakout_high"]), float(row["breakout_low"])
    previous_close = float(row["previous_close"])
    close_location = float(row["close_location"])
    low, high = float(row["low"]), float(row["high"])
    crossed_up = float(row["previous_ema9"]) <= float(row["previous_ema21"]) and ema9 > ema21
    crossed_down = float(row["previous_ema9"]) >= float(row["previous_ema21"]) and ema9 < ema21
    breakout_up = (
        previous_close <= breakout_high < close
        and close - breakout_high <= signal.atr * 0.35
    )
    breakout_down = (
        previous_close >= breakout_low > close
        and breakout_low - close <= signal.atr * 0.35
    )
    pullback_up = (
        previous_close <= float(row["previous_ema9"])
        and low <= ema21 + signal.atr * 0.20
        and close > ema9
    )
    pullback_down = (
        previous_close >= float(row["previous_ema9"])
        and high >= ema21 - signal.atr * 0.20
        and close < ema9
    )
    sr_retest_up = previous_close > breakout_high and low <= breakout_high + signal.atr * 0.15 < close
    sr_retest_down = previous_close < breakout_low and high >= breakout_low - signal.atr * 0.15 > close
    bb_rejection_up = low <= float(row["bb_lower"]) and close > float(row["bb_lower"])
    bb_rejection_down = high >= float(row["bb_upper"]) and close < float(row["bb_upper"])
    buy_setups = {
        "breakout": breakout_up,
        "MA crossover": crossed_up,
        "EMA pullback": pullback_up,
        "support retest": sr_retest_up,
        "Bollinger rejection": bb_rejection_up,
    }
    sell_setups = {
        "breakout": breakout_down,
        "MA crossover": crossed_down,
        "EMA pullback": pullback_down,
        "resistance retest": sr_retest_down,
        "Bollinger rejection": bb_rejection_down,
    }
    buy_votes = sum(buy_setups.values())
    sell_votes = sum(sell_setups.values())
    # The multi-setup ensemble was tuned and validated specifically on XAUUSD.
    # Broker suffixes are accepted, while BTC/forex keep the simpler setup so
    # gold-specific behaviour is not silently applied to another instrument.
    xau_ensemble = "XAU" in symbol.upper() or "GOLD" in symbol.upper()
    combined_buy = (
        breakout_up or buy_votes >= 2
        if xau_ensemble
        else breakout_up or crossed_up
    )
    combined_sell = (
        breakout_down or sell_votes >= 2
        if xau_ensemble
        else breakout_down or crossed_down
    )
    common = {
        "volume confirmed": float(row["volume_ratio"]) >= p.min_volume_ratio,
        "volatility tradable": 0.00003 <= atr_ratio <= p.max_atr_ratio,
        "breakout candle quality": p.min_body_atr <= body_atr <= 1.20,
    }
    buy = {
        "macro trend up": ema50 > ema200,
        "EMA21 rising": float(row["ema21_slope"]) > 0,
        "combined bullish setup": combined_buy,
        "setup strength confirmed": crossed_up or pullback_up or float(row["trend_strength"]) >= p.min_trend_strength,
        "non-breakout volume surge": breakout_up or float(row["volume_ratio"]) >= p.min_volume_ratio + 0.15,
        "price above aligned MAs": close > ema9 > ema21 > ema50,
        "strong bullish close": close_location >= 0.70,
        "RSI bullish": 56 <= rsi <= 70,
        "MACD positive": macd > 0,
        **common,
    }
    sell = {
        "macro trend down": ema50 < ema200,
        "EMA21 falling": float(row["ema21_slope"]) < 0,
        "combined bearish setup": combined_sell,
        "setup strength confirmed": crossed_down or pullback_down or float(row["trend_strength"]) >= p.min_trend_strength,
        "non-breakout volume surge": breakout_down or float(row["volume_ratio"]) >= p.min_volume_ratio + 0.15,
        "price below aligned MAs": close < ema9 < ema21 < ema50,
        "strong bearish close": close_location <= 0.30,
        "RSI bearish": 30 <= rsi <= 44,
        "MACD negative": macd < 0,
        **common,
    }
    if all(buy.values()):
        active = [name for name, enabled in buy_setups.items() if enabled]
        signal.action, signal.reasons = "BUY", [f"setups: {', '.join(active)}", *buy]
    elif all(sell.values()):
        active = [name for name, enabled in sell_setups.items() if enabled]
        signal.action, signal.reasons = "SELL", [f"setups: {', '.join(active)}", *sell]
    else:
        signal.reasons = [f"BUY {sum(buy.values())}/{len(buy)}", f"SELL {sum(sell.values())}/{len(sell)}"]
    return signal


def build_m5_plan(direction: str, entry: float, atr_value: float, balance: float, symbol: str, risk_percent: float = 0.5) -> ScalpingPlan:
    p = get_m5_preset(symbol)
    return build_scalping_plan(direction, entry, atr_value, balance, symbol, risk_percent, p.stop_atr, p.risk_reward)
