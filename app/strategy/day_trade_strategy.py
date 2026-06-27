"""Standalone multi-market intraday momentum-breakout strategy.

This module is intentionally independent from the existing live signal engine.
It combines trend, momentum, volatility, breakout, and volume confirmation so
the same rules can be evaluated on crypto, metals, forex, and indices.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from app.ml.feature_engineering import build_features
from app.strategy.risk_manager import calculate_lot


@dataclass
class DayTradeSignal:
    action: str = "HOLD"
    price: float = 0.0
    atr: float = 0.0
    reasons: list[str] = field(default_factory=list)


@dataclass
class DayTradePlan:
    direction: str
    entry: float
    stop_loss: float
    take_profit: float
    lot: float
    risk_money: float
    rr: float


@dataclass(frozen=True)
class DayTradePreset:
    name: str
    breakout_period: int
    min_volume_ratio: float
    min_trend_strength: float
    buy_rsi_min: float
    buy_rsi_max: float
    sell_rsi_min: float
    sell_rsi_max: float
    stop_atr: float
    risk_reward: float


def get_day_trade_preset(symbol: str, profile: str = "auto") -> DayTradePreset:
    """Return conservative instrument-family parameters without changing rules."""
    name = profile.lower()
    upper = symbol.upper()
    if name == "auto":
        name = "crypto" if any(x in upper for x in ("BTC", "ETH", "SOL", "XRP")) else "metals" if any(x in upper for x in ("XAU", "XAG")) else "standard"
    if name == "crypto":
        return DayTradePreset("crypto", 48, 1.10, 0.35, 55, 70, 30, 45, 1.50, 2.20)
    if name == "metals":
        return DayTradePreset("metals", 20, 0.80, 0.15, 52, 72, 28, 48, 1.25, 1.80)
    return DayTradePreset("standard", 30, 1.00, 0.20, 53, 70, 30, 47, 1.35, 2.00)


def prepare_day_trade_features(
    df: pd.DataFrame,
    breakout_period: int = 20,
) -> pd.DataFrame:
    """Add standard indicators and prior-bar Donchian breakout levels."""
    out = build_features(df)
    out["breakout_high"] = out["high"].shift(1).rolling(breakout_period).max()
    out["breakout_low"] = out["low"].shift(1).rolling(breakout_period).min()
    out["atr_ratio"] = out["atr"] / out["close"].replace(0, pd.NA)
    out["trend_strength"] = (
        (out["ema20"] - out["ema50"]).abs()
        / out["atr"].replace(0, pd.NA)
    )
    volume_avg = out["volume_avg"].replace(0, pd.NA)
    out["volume_ratio"] = (out["volume"] / volume_avg).fillna(1.0)
    return out


def generate_day_trade_signal(
    row: pd.Series,
    spread_points: float,
    max_spread_points: float,
    preset: DayTradePreset | None = None,
) -> DayTradeSignal:
    """Generate BUY/SELL only when every independent confirmation aligns."""
    signal = DayTradeSignal(
        price=float(row.get("close", 0.0) or 0.0),
        atr=float(row.get("atr", 0.0) or 0.0),
    )
    required = [
        "ema20", "ema50", "ema200", "rsi", "macd_hist", "breakout_high",
        "breakout_low", "atr_ratio", "trend_strength", "volume_ratio",
    ]
    if any(pd.isna(row.get(name)) for name in required):
        signal.reasons = ["indicators not ready"]
        return signal
    if spread_points > max_spread_points:
        signal.reasons = [f"spread too high ({spread_points:.1f})"]
        return signal

    close = signal.price
    ema20, ema50, ema200 = (float(row[x]) for x in ("ema20", "ema50", "ema200"))
    rsi = float(row["rsi"])
    macd_hist = float(row["macd_hist"])
    atr_ratio = float(row["atr_ratio"])
    trend_strength = float(row["trend_strength"])
    volume_ratio = float(row["volume_ratio"])
    body = float(row.get("candle_body", 0.0) or 0.0)

    preset = preset or get_day_trade_preset("", "standard")
    common = {
        "volatility active": 0.0004 <= atr_ratio <= 0.08,
        "trend has strength": trend_strength >= preset.min_trend_strength,
        "volume confirmed": volume_ratio >= preset.min_volume_ratio,
        "candle not exhausted": signal.atr > 0 and body <= signal.atr * 2.0,
    }
    buy = {
        "EMA20>EMA50>EMA200": ema20 > ema50 > ema200,
        "Donchian breakout up": close > float(row["breakout_high"]),
        "RSI buy zone": preset.buy_rsi_min <= rsi <= preset.buy_rsi_max,
        "MACD momentum up": macd_hist > 0,
        **common,
    }
    sell = {
        "EMA20<EMA50<EMA200": ema20 < ema50 < ema200,
        "Donchian breakout down": close < float(row["breakout_low"]),
        "RSI sell zone": preset.sell_rsi_min <= rsi <= preset.sell_rsi_max,
        "MACD momentum down": macd_hist < 0,
        **common,
    }
    if all(buy.values()):
        signal.action = "BUY"
        signal.reasons = list(buy)
    elif all(sell.values()):
        signal.action = "SELL"
        signal.reasons = list(sell)
    else:
        signal.reasons = [
            f"BUY {sum(buy.values())}/{len(buy)}",
            f"SELL {sum(sell.values())}/{len(sell)}",
        ]
    return signal


def build_day_trade_plan(
    direction: str,
    entry: float,
    atr_value: float,
    balance: float,
    symbol: str,
    risk_percent: float = 1.0,
    stop_atr: float = 1.25,
    risk_reward: float = 1.8,
) -> DayTradePlan:
    """Build an ATR-normalized plan suitable for instruments of any price."""
    direction = direction.upper()
    distance = max(atr_value * stop_atr, entry * 0.0004)
    if direction == "BUY":
        stop_loss = entry - distance
        take_profit = entry + distance * risk_reward
    else:
        stop_loss = entry + distance
        take_profit = entry - distance * risk_reward
    lot = calculate_lot(balance, risk_percent, entry, stop_loss, symbol)
    return DayTradePlan(
        direction=direction,
        entry=entry,
        stop_loss=stop_loss,
        take_profit=take_profit,
        lot=lot,
        risk_money=balance * risk_percent / 100.0,
        rr=risk_reward,
    )
