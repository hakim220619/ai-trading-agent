"""Support/resistance, swing points, breakout and retest detection."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class SRLevels:
    """Container for support/resistance analysis results."""

    support: float | None = None
    resistance: float | None = None
    swing_highs: list[float] = field(default_factory=list)
    swing_lows: list[float] = field(default_factory=list)
    breakout: str | None = None  # "up" | "down" | None
    retest: bool = False
    structure_trend: str = "ranging"  # bullish | bearish | ranging
    bos: str | None = None  # bullish | bearish | None
    bos_level: float | None = None
    choch: str | None = None  # bullish | bearish | None
    choch_level: float | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "support": self.support,
            "resistance": self.resistance,
            "swing_highs": self.swing_highs[-5:],
            "swing_lows": self.swing_lows[-5:],
            "breakout": self.breakout,
            "retest": self.retest,
            "structure_trend": self.structure_trend,
            "bos": self.bos,
            "bos_level": self.bos_level,
            "choch": self.choch,
            "choch_level": self.choch_level,
        }


def find_swings(df: pd.DataFrame, left: int = 3, right: int = 3) -> tuple[list[float], list[float]]:
    """Detect fractal swing highs and lows.

    A swing high is a bar whose high is greater than ``left`` bars before and
    ``right`` bars after it; swing low is the symmetric opposite.
    """
    highs: list[float] = []
    lows: list[float] = []
    n = len(df)
    if n < left + right + 1:
        return highs, lows

    high = df["high"].to_numpy()
    low = df["low"].to_numpy()

    for i in range(left, n - right):
        window_high = high[i - left : i + right + 1]
        window_low = low[i - left : i + right + 1]
        if high[i] == window_high.max() and (window_high == high[i]).sum() == 1:
            highs.append(float(high[i]))
        if low[i] == window_low.min() and (window_low == low[i]).sum() == 1:
            lows.append(float(low[i]))
    return highs, lows


def detect_levels(df: pd.DataFrame, left: int = 3, right: int = 3, tolerance: float = 0.0015) -> SRLevels:
    """Compute nearest support/resistance and breakout/retest state.

    ``tolerance`` is a fraction of price used to decide "near" / retest.
    """
    result = SRLevels()
    if df is None or df.empty or len(df) < left + right + 2:
        return result

    highs, lows = find_swings(df, left, right)
    result.swing_highs = highs
    result.swing_lows = lows

    price = float(df["close"].iloc[-1])

    # Market structure uses the last two confirmed swing highs and lows.
    # HH + HL = bullish; LH + LL = bearish; mixed structure = ranging.
    if len(highs) >= 2 and len(lows) >= 2:
        higher_high = highs[-1] > highs[-2]
        higher_low = lows[-1] > lows[-2]
        lower_high = highs[-1] < highs[-2]
        lower_low = lows[-1] < lows[-2]
        if higher_high and higher_low:
            result.structure_trend = "bullish"
        elif lower_high and lower_low:
            result.structure_trend = "bearish"

    # Nearest resistance = lowest swing high above price.
    above = [h for h in highs if h > price]
    result.resistance = min(above) if above else None

    # Nearest support = highest swing low below price.
    below = [low for low in lows if low < price]
    result.support = max(below) if below else None

    # Break of Structure (continuation) and Change of Character (reversal).
    # Only close-to-close crossings count, preventing the same break from being
    # reported repeatedly on every following candle.
    if len(df) >= 2:
        prev_close = float(df["close"].iloc[-2])
        latest_high = highs[-1] if highs else None
        latest_low = lows[-1] if lows else None
        broke_high = latest_high is not None and prev_close <= latest_high < price
        broke_low = latest_low is not None and prev_close >= latest_low > price
        if broke_high:
            result.breakout = "up"
            if result.structure_trend == "bearish":
                result.choch = "bullish"
                result.choch_level = latest_high
            else:
                result.bos = "bullish"
                result.bos_level = latest_high
        elif broke_low:
            result.breakout = "down"
            if result.structure_trend == "bullish":
                result.choch = "bearish"
                result.choch_level = latest_low
            else:
                result.bos = "bearish"
                result.bos_level = latest_low

    # Retest: price currently hugging a level within tolerance.
    tol = price * tolerance
    if result.support is not None and abs(price - result.support) <= tol:
        result.retest = True
    if result.resistance is not None and abs(price - result.resistance) <= tol:
        result.retest = True

    return result


def distance_to_support(df: pd.DataFrame, levels: SRLevels) -> float | None:
    """Fractional distance of current price above its support (None if N/A)."""
    if levels.support is None or df.empty:
        return None
    price = float(df["close"].iloc[-1])
    return abs(price - levels.support) / price


def distance_to_resistance(df: pd.DataFrame, levels: SRLevels) -> float | None:
    """Fractional distance of current price below its resistance (None if N/A)."""
    if levels.resistance is None or df.empty:
        return None
    price = float(df["close"].iloc[-1])
    return abs(levels.resistance - price) / price


def is_near_support(df: pd.DataFrame, levels: SRLevels, tolerance: float = 0.003) -> bool:
    d = distance_to_support(df, levels)
    return d is not None and d <= tolerance


def is_near_resistance(df: pd.DataFrame, levels: SRLevels, tolerance: float = 0.003) -> bool:
    d = distance_to_resistance(df, levels)
    return d is not None and d <= tolerance
