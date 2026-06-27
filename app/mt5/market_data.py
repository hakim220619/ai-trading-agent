"""Fetch OHLCV candle data from MetaTrader 5."""
from __future__ import annotations

import pandas as pd

from app.config import settings
from app.mt5.connection import MT5_AVAILABLE, connection, mt5, timeframe_map
from app.utils.logger import logger


def get_candles(
    symbol: str | None = None,
    timeframe: str = "M5",
    count: int | None = None,
) -> pd.DataFrame:
    """Return a DataFrame of OHLCV candles for ``symbol`` / ``timeframe``.

    Columns: time, open, high, low, close, tick_volume, spread, real_volume.
    A ``volume`` alias column is added (tick_volume) for convenience.
    Returns an empty DataFrame on failure.
    """
    symbol = symbol or settings.symbol
    count = count or settings.candles

    if not MT5_AVAILABLE or not connection.ensure_connected():
        logger.error("get_candles: MT5 not connected.")
        return pd.DataFrame()

    tf_map = timeframe_map()
    if timeframe not in tf_map:
        logger.error("Unknown timeframe '{}'. Valid: {}", timeframe, list(tf_map))
        return pd.DataFrame()

    if not mt5.symbol_select(symbol, True):
        logger.error("Cannot select symbol {}", symbol)
        return pd.DataFrame()

    rates = mt5.copy_rates_from_pos(symbol, tf_map[timeframe], 0, count)
    if rates is None or len(rates) == 0:
        logger.error("copy_rates_from_pos returned no data: {}", mt5.last_error())
        return pd.DataFrame()

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    if "tick_volume" in df.columns:
        df["volume"] = df["tick_volume"]
    else:
        df["volume"] = 0
    logger.debug("Fetched {} candles {} {}", len(df), symbol, timeframe)
    return df


def get_multi_timeframe(
    symbol: str | None = None,
    timeframes: list[str] | None = None,
    count: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Return a dict mapping each timeframe to its candle DataFrame."""
    symbol = symbol or settings.symbol
    timeframes = timeframes or settings.timeframes
    return {tf: get_candles(symbol, tf, count) for tf in timeframes}


def get_current_tick(symbol: str | None = None) -> dict[str, float] | None:
    """Return current bid/ask/last/time for ``symbol`` (None on failure)."""
    symbol = symbol or settings.symbol
    if not MT5_AVAILABLE or not connection.ensure_connected():
        return None
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        logger.error("symbol_info_tick({}) failed: {}", symbol, mt5.last_error())
        return None
    return {
        "bid": tick.bid,
        "ask": tick.ask,
        "last": tick.last,
        "time": tick.time,
    }


def load_candles_csv(path: str) -> pd.DataFrame:
    """Load candles from a CSV file (used for backtesting / training offline).

    Expects at least: time, open, high, low, close, volume (or tick_volume).
    """
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
    if "volume" not in df.columns and "tick_volume" in df.columns:
        df["volume"] = df["tick_volume"]
    if "spread" not in df.columns:
        df["spread"] = 0
    logger.info("Loaded {} rows from {}", len(df), path)
    return df
