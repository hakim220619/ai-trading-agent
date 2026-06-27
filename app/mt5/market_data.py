"""Fetch OHLCV candle data from MetaTrader 5."""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

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


def get_candles_range(
    date_from: datetime,
    date_to: datetime,
    symbol: str | None = None,
    timeframe: str = "M5",
) -> pd.DataFrame:
    """Return MT5 candles inside a UTC date range."""
    symbol = symbol or settings.symbol
    timeframe = timeframe.upper()
    if not MT5_AVAILABLE or not connection.ensure_connected():
        logger.error("get_candles_range: MT5 not connected.")
        return pd.DataFrame()
    tf_map = timeframe_map()
    if timeframe not in tf_map:
        logger.error("Unknown timeframe '{}'. Valid: {}", timeframe, list(tf_map))
        return pd.DataFrame()
    if not mt5.symbol_select(symbol, True):
        logger.error("Cannot select symbol {}", symbol)
        return pd.DataFrame()
    # Some terminals reject a single request near/above 100k bars. Fetching in
    # bounded chunks works for multi-year M5 history and remains safe for H1.
    chunks: list[pd.DataFrame] = []
    cursor = date_from
    while cursor < date_to:
        chunk_end = min(cursor + timedelta(days=60), date_to)
        rates = mt5.copy_rates_range(
            symbol,
            tf_map[timeframe],
            int(cursor.timestamp()),
            int(chunk_end.timestamp()),
        )
        if rates is not None and len(rates) > 0:
            chunks.append(pd.DataFrame(rates))
        else:
            logger.debug(
                "No ranged candles for {} to {}: {}",
                cursor,
                chunk_end,
                mt5.last_error(),
            )
        cursor = chunk_end + timedelta(seconds=1)
    if not chunks:
        logger.error("copy_rates_range returned no data: {}", mt5.last_error())
        return pd.DataFrame()
    df = pd.concat(chunks, ignore_index=True).drop_duplicates(subset=["time"])
    df = df.sort_values("time").reset_index(drop=True)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_localize(None)
    df["volume"] = df["tick_volume"] if "tick_volume" in df.columns else 0
    logger.info("Fetched {} ranged candles {} {}", len(df), symbol, timeframe)
    return df


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


def export_candles_csv(
    symbol: str | None = None,
    timeframe: str = "M5",
    count: int = 10_000,
    path: str | None = None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
) -> dict[str, object]:
    """Fetch historical MT5 candles and save them as a training-ready CSV."""
    symbol = symbol or settings.symbol
    timeframe = timeframe.upper()
    if date_from is not None:
        if date_to is None:
            raise ValueError("date_to is required when date_from is supplied")
        df = get_candles_range(date_from, date_to, symbol=symbol, timeframe=timeframe)
    else:
        df = get_candles(symbol=symbol, timeframe=timeframe, count=count)
    if df.empty:
        raise RuntimeError(f"No MT5 candle data returned for {symbol} {timeframe}")

    safe_symbol = "".join(c for c in symbol if c.isalnum() or c in "-_.")
    output = Path(path) if path else Path("data") / f"{safe_symbol}_{timeframe}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    logger.success("Exported {} candles to {}", len(df), output)
    result: dict[str, object] = {
        "path": str(output),
        "rows": len(df),
        "symbol": symbol,
        "timeframe": timeframe,
        "first_time": str(df["time"].iloc[0]) if "time" in df else None,
        "last_time": str(df["time"].iloc[-1]) if "time" in df else None,
    }
    if date_from is not None and date_to is not None:
        result["requested_from"] = date_from.isoformat()
        result["requested_to"] = date_to.isoformat()
    return result
