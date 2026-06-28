"""Standalone M1 trend-pullback scalping strategy for all liquid markets."""
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
class M1Preset:
    stop_atr: float
    risk_reward: float
    min_volume_ratio: float
    max_atr_ratio: float
    pullback_atr: float


def get_m1_preset(symbol: str) -> M1Preset:
    family = market_family(symbol)
    if family == "crypto":
        return M1Preset(1.35, 1.35, 1.05, 0.025, 0.55)
    if family == "metals":
        return M1Preset(1.15, 1.30, 1.10, 0.012, 0.45)
    return M1Preset(1.05, 1.25, 1.00, 0.008, 0.40)


def prepare_m1_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add M1 setups plus the latest fully-closed H1 trend context."""
    out = build_features(df)
    out["ema5"] = out["close"].ewm(span=5, adjust=False).mean()
    out["ema13"] = out["close"].ewm(span=13, adjust=False).mean()
    out["atr_ratio"] = out["atr"] / out["close"].replace(0, pd.NA)
    out["volume_ratio"] = out["volume"] / out["volume_avg"].replace(0, pd.NA)
    out["previous_close"] = out["close"].shift(1)
    out["previous_ema5"] = out["ema5"].shift(1)
    out["previous_ema13"] = out["ema13"].shift(1)
    out["breakout_high"] = out["high"].shift(1).rolling(12).max()
    out["breakout_low"] = out["low"].shift(1).rolling(12).min()
    out["prior_breakout_high"] = out["breakout_high"].shift(1)
    out["prior_breakout_low"] = out["breakout_low"].shift(1)
    out["micro_trend_strength"] = (out["ema5"] - out["ema13"]).abs() / out["atr"].replace(0, pd.NA)

    # H1 values become visible to M1 only at the end of their hour. This
    # prevents an incomplete hourly candle leaking future data into entries.
    if "time" in out and len(out) > 0:
        timed = out[["time", "open", "high", "low", "close", "volume"]].copy()
        timed["time"] = pd.to_datetime(timed["time"], errors="coerce")
        source = timed.dropna(subset=["time"]).set_index("time")

        def closed_context(rule: str, delay: pd.Timedelta, prefix: str) -> pd.DataFrame:
            context_bars = (
                source.resample(rule)
            .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
            .dropna(subset=["open", "high", "low", "close"])
            .reset_index()
            )
            context_bars = build_features(context_bars)
            context_bars["available_time"] = context_bars["time"] + delay
            return context_bars[["available_time", "ema20", "ema50", "ema200", "rsi", "macd_hist"]].rename(
                columns={name: f"{prefix}_{name}" for name in ("ema20", "ema50", "ema200", "rsi", "macd_hist")}
            )

        contexts = [
            closed_context("5min", pd.Timedelta(minutes=5), "m5"),
            closed_context("1h", pd.Timedelta(hours=1), "h1"),
        ]
        base = out.reset_index().rename(columns={"index": "_original_index"})
        base["time"] = pd.to_datetime(base["time"], errors="coerce")
        for context in contexts:
            base = pd.merge_asof(
                base.sort_values("time"), context.sort_values("available_time"),
                left_on="time", right_on="available_time", direction="backward",
            ).drop(columns=["available_time"])
        out = base.sort_values("_original_index").drop(columns=["_original_index"])
    return out


def generate_m1_signal(
    row: pd.Series,
    symbol: str,
    spread_points: float,
    max_spread_points: float,
    min_setup_confirmations: int = 3,
) -> ScalpingSignal:
    """Trade a shallow pullback in a strong micro-trend on a closed M1 bar."""
    signal = ScalpingSignal(
        price=float(row.get("close", 0) or 0),
        atr=float(row.get("atr", 0) or 0),
    )
    required = (
        "ema5", "ema13", "ema50", "rsi", "macd_hist", "atr_ratio",
        "volume_ratio", "low", "high", "previous_ema5", "previous_ema13",
        "breakout_high", "breakout_low", "h1_ema20", "h1_ema50",
        "h1_ema200", "h1_rsi", "h1_macd_hist", "m5_ema20", "m5_ema50",
        "m5_ema200", "m5_rsi", "m5_macd_hist", "prior_breakout_high",
        "prior_breakout_low", "micro_trend_strength",
    )
    if any(pd.isna(row.get(name)) for name in required) or signal.atr <= 0:
        signal.reasons = ["indicators not ready"]
        return signal
    if spread_points > max_spread_points:
        signal.reasons = [f"spread too high ({spread_points:.1f})"]
        return signal
    if min_setup_confirmations not in (1, 2, 3, 4):
        raise ValueError("min_setup_confirmations must be between 1 and 4")

    p = get_m1_preset(symbol)
    close, ema5, ema13, ema50 = (float(row[x]) for x in ("close", "ema5", "ema13", "ema50"))
    rsi, macd, atr_ratio = (float(row[x]) for x in ("rsi", "macd_hist", "atr_ratio"))
    volume_ok = float(row["volume_ratio"]) >= p.min_volume_ratio
    volatility_ok = 0.00002 <= atr_ratio <= p.max_atr_ratio
    candle_ok = float(row.get("candle_body", 0) or 0) <= signal.atr * 1.5
    body_atr = float(row.get("candle_body", 0) or 0) / signal.atr
    candle_range = max(float(row["high"]) - float(row["low"]), 1e-12)
    close_location = (close - float(row["low"])) / candle_range
    crossed_up = float(row["previous_ema5"]) <= float(row["previous_ema13"]) and ema5 > ema13
    crossed_down = float(row["previous_ema5"]) >= float(row["previous_ema13"]) and ema5 < ema13
    retest_up = (
        float(row["previous_close"]) > float(row["prior_breakout_high"])
        and float(row["low"]) <= float(row["prior_breakout_high"]) + signal.atr * 0.15
        and close > float(row["prior_breakout_high"])
    )
    retest_down = (
        float(row["previous_close"]) < float(row["prior_breakout_low"])
        and float(row["high"]) >= float(row["prior_breakout_low"]) - signal.atr * 0.15
        and close < float(row["prior_breakout_low"])
    )
    pullback_up = float(row["low"]) <= ema5 + signal.atr * p.pullback_atr and close > ema5
    pullback_down = float(row["high"]) >= ema5 - signal.atr * p.pullback_atr and close < ema5
    bb_up = float(row["low"]) <= float(row["bb_lower"]) and close > float(row["bb_lower"])
    bb_down = float(row["high"]) >= float(row["bb_upper"]) and close < float(row["bb_upper"])
    h1_buy = float(row["h1_ema20"]) > float(row["h1_ema50"]) > float(row["h1_ema200"]) and float(row["h1_macd_hist"]) > 0 and float(row["h1_rsi"]) >= 50
    h1_sell = float(row["h1_ema20"]) < float(row["h1_ema50"]) < float(row["h1_ema200"]) and float(row["h1_macd_hist"]) < 0 and float(row["h1_rsi"]) <= 50
    m5_buy = float(row["m5_ema20"]) > float(row["m5_ema50"]) > float(row["m5_ema200"]) and float(row["m5_macd_hist"]) > 0 and float(row["m5_rsi"]) >= 50
    m5_sell = float(row["m5_ema20"]) < float(row["m5_ema50"]) < float(row["m5_ema200"]) and float(row["m5_macd_hist"]) < 0 and float(row["m5_rsi"]) <= 50
    buy_setups = [crossed_up, retest_up, pullback_up, bb_up]
    sell_setups = [crossed_down, retest_down, pullback_down, bb_down]
    buy = {
        "closed H1 trend up": h1_buy,
        "closed M5 trend up": m5_buy,
        "micro trend up": close > ema5 > ema13 > ema50,
        "market not sideways": float(row["micro_trend_strength"]) >= 0.10,
        f"{min_setup_confirmations} M1 setups agree": sum(buy_setups) >= min_setup_confirmations,
        "RSI bullish, not exhausted": 52 <= rsi <= 72,
        "MACD positive": macd > 0,
        "volume active": volume_ok,
        "volatility tradable": volatility_ok,
        "candle controlled": candle_ok,
        "strong bullish candle": body_atr >= 0.20 and close_location >= 0.65,
    }
    sell = {
        "closed H1 trend down": h1_sell,
        "closed M5 trend down": m5_sell,
        "micro trend down": close < ema5 < ema13 < ema50,
        "market not sideways": float(row["micro_trend_strength"]) >= 0.10,
        f"{min_setup_confirmations} M1 setups agree": sum(sell_setups) >= min_setup_confirmations,
        "RSI bearish, not exhausted": 28 <= rsi <= 48,
        "MACD negative": macd < 0,
        "volume active": volume_ok,
        "volatility tradable": volatility_ok,
        "candle controlled": candle_ok,
        "strong bearish candle": body_atr >= 0.20 and close_location <= 0.35,
    }
    if all(buy.values()):
        signal.action, signal.reasons = "BUY", list(buy)
    elif all(sell.values()):
        signal.action, signal.reasons = "SELL", list(sell)
    else:
        signal.reasons = [f"BUY {sum(buy.values())}/{len(buy)}", f"SELL {sum(sell.values())}/{len(sell)}"]
    return signal


def build_m1_plan(direction: str, entry: float, atr_value: float, balance: float, symbol: str, risk_percent: float = 0.5) -> ScalpingPlan:
    p = get_m1_preset(symbol)
    return build_scalping_plan(direction, entry, atr_value, balance, symbol, risk_percent, p.stop_atr, p.risk_reward)
