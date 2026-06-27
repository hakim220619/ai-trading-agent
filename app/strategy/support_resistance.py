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

    def to_dict(self) -> dict[str, object]:
        return {
            "support": self.support,
            "resistance": self.resistance,
            "swing_highs": self.swing_highs[-5:],
            "swing_lows": self.swing_lows[-5:],
            "breakout": self.breakout,
            "retest": self.retest,
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

    # Nearest resistance = lowest swing high above price.
    above = [h for h in highs if h > price]
    result.resistance = min(above) if above else (max(highs) if highs else None)

    # Nearest support = highest swing low below price.
    below = [low for low in lows if low < price]
    result.support = max(below) if below else (min(lows) if lows else None)

    # Breakout: previous bar closed beyond a recent level, current confirms.
    if len(df) >= 2:
        prev_close = float(df["close"].iloc[-2])
        if result.resistance and prev_close <= result.resistance < price:
            result.breakout = "up"
        elif result.support and prev_close >= result.support > price:
            result.breakout = "down"

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
