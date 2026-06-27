"""Technical indicator calculations.

Implemented directly on top of pandas/numpy so the project does not depend on a
specific pandas-ta / numpy ABI combination. All functions are vectorised and
return new columns on a copy of the input DataFrame.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder's smoothing)."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)


def macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD line, signal line and histogram."""
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def bollinger_bands(
    series: pd.Series, period: int = 20, std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger middle, upper and lower bands."""
    middle = series.rolling(period).mean()
    deviation = series.rolling(period).std(ddof=0)
    upper = middle + std * deviation
    lower = middle - std * deviation
    return middle, upper, lower


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with all standard indicator columns added.

    Adds: ema20, ema50, ema200, rsi, macd, macd_signal, macd_hist, atr,
    bb_middle, bb_upper, bb_lower, candle_body, upper_wick, lower_wick,
    volume_avg.
    """
    if df is None or df.empty:
        return df

    out = df.copy()
    close = out["close"]

    out["ema20"] = ema(close, 20)
    out["ema50"] = ema(close, 50)
    out["ema200"] = ema(close, 200)
    out["rsi"] = rsi(close, 14)

    macd_line, macd_signal, macd_hist = macd(close)
    out["macd"] = macd_line
    out["macd_signal"] = macd_signal
    out["macd_hist"] = macd_hist

    out["atr"] = atr(out, 14)

    bb_mid, bb_up, bb_low = bollinger_bands(close, 20, 2.0)
    out["bb_middle"] = bb_mid
    out["bb_upper"] = bb_up
    out["bb_lower"] = bb_low

    body = (out["close"] - out["open"]).abs()
    out["candle_body"] = body
    out["upper_wick"] = out["high"] - out[["close", "open"]].max(axis=1)
    out["lower_wick"] = out[["close", "open"]].min(axis=1) - out["low"]

    vol = out["volume"] if "volume" in out.columns else pd.Series(0, index=out.index)
    out["volume_avg"] = vol.rolling(20).mean()

    return out


def latest_indicator_snapshot(df: pd.DataFrame) -> dict[str, float]:
    """Return the last row's indicator values as a plain dict of floats."""
    if df is None or df.empty:
        return {}
    row = df.iloc[-1]
    keys = [
        "open", "high", "low", "close", "volume",
        "ema20", "ema50", "ema200", "rsi",
        "macd", "macd_signal", "macd_hist", "atr",
        "bb_middle", "bb_upper", "bb_lower",
        "candle_body", "upper_wick", "lower_wick", "volume_avg",
    ]
    snap: dict[str, float] = {}
    for k in keys:
        if k in row and pd.notna(row[k]):
            snap[k] = float(row[k])
    return snap
