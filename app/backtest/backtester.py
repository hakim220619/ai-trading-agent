"""Event-driven backtester over historical candles.

Replays candles bar-by-bar, generating signals with the same rule+ML engine the
live bot uses, then simulates SL/TP fills on subsequent bars.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from app.config import settings
from app.ml.feature_engineering import FEATURE_COLUMNS, build_features
from app.strategy.risk_manager import build_trade_plan
from app.strategy.signal_generator import generate_signal
from app.utils.logger import logger


@dataclass
class Trade:
    """A single simulated trade."""

    direction: str
    entry_time: object
    entry: float
    sl: float
    tp: float
    lot: float
    exit_time: object | None = None
    exit: float | None = None
    pnl: float = 0.0
    result: str = "open"  # win | loss | open

    def to_dict(self) -> dict[str, object]:
        return {
            "direction": self.direction,
            "entry_time": str(self.entry_time),
            "entry": round(self.entry, 5),
            "sl": round(self.sl, 5),
            "tp": round(self.tp, 5),
            "lot": self.lot,
            "exit_time": str(self.exit_time) if self.exit_time else None,
            "exit": round(self.exit, 5) if self.exit else None,
            "pnl": round(self.pnl, 2),
            "result": self.result,
        }


@dataclass
class BacktestResult:
    """Aggregate backtest output."""

    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    start_balance: float = 0.0
    end_balance: float = 0.0


def _pnl_per_price_unit(lot: float) -> float:
    """Approx PnL per 1.0 price move per lot (XAUUSD ~100 oz contract)."""
    # tick_size 0.01 -> tick_value 1.0 => 1.0 price move = 100 * tick_value * lot
    return 100.0 * lot


def run_backtest(
    df: pd.DataFrame,
    start_balance: float = 1000.0,
    warmup: int = 200,
    max_hold: int = 96,
) -> BacktestResult:
    """Run a vectorised-ish backtest.

    ``warmup`` bars are skipped so indicators are valid. Each open trade is held
    until SL/TP hits or ``max_hold`` bars elapse. Only one position at a time.
    """
    if df is None or len(df) <= warmup + 10:
        raise ValueError("Not enough data to backtest.")

    df = build_features(df).reset_index(drop=True)
    result = BacktestResult(start_balance=start_balance)
    balance = start_balance
    open_trade: Trade | None = None
    open_idx = 0

    for i in range(warmup, len(df) - 1):
        window = df.iloc[: i + 1]
        row = df.iloc[i]

        # --- manage open trade ---
        if open_trade is not None:
            high = df.iloc[i]["high"]
            low = df.iloc[i]["low"]
            closed = False
            if open_trade.direction == "BUY":
                if low <= open_trade.sl:
                    open_trade.exit = open_trade.sl
                    open_trade.result = "loss"
                    closed = True
                elif high >= open_trade.tp:
                    open_trade.exit = open_trade.tp
                    open_trade.result = "win"
                    closed = True
            else:  # SELL
                if high >= open_trade.sl:
                    open_trade.exit = open_trade.sl
                    open_trade.result = "loss"
                    closed = True
                elif low <= open_trade.tp:
                    open_trade.exit = open_trade.tp
                    open_trade.result = "win"
                    closed = True

            if not closed and (i - open_idx) >= max_hold:
                open_trade.exit = row["close"]
                open_trade.result = "win" if _trade_pnl(open_trade) > 0 else "loss"
                closed = True

            if closed:
                open_trade.exit_time = row["time"] if "time" in row else i
                open_trade.pnl = _trade_pnl(open_trade)
                balance += open_trade.pnl
                result.trades.append(open_trade)
                open_trade = None

        result.equity_curve.append(balance)

        # --- look for new entry only when flat ---
        if open_trade is None:
            sig = generate_signal(window, features=_row_features(row))
            if sig.action in ("BUY", "SELL"):
                plan = build_trade_plan(
                    direction=sig.action,
                    entry=float(row["close"]),
                    atr_value=float(row.get("atr", 0.0) or 0.0),
                    balance=balance,
                    swing_high=sig.levels.get("resistance"),  # type: ignore[arg-type]
                    swing_low=sig.levels.get("support"),       # type: ignore[arg-type]
                )
                open_trade = Trade(
                    direction=sig.action,
                    entry_time=row["time"] if "time" in row else i,
                    entry=plan.entry,
                    sl=plan.stop_loss,
                    tp=plan.take_profit,
                    lot=plan.lot,
                )
                open_idx = i

    result.end_balance = balance
    logger.info(
        "Backtest done: {} trades, balance {:.2f} -> {:.2f}",
        len(result.trades),
        start_balance,
        balance,
    )
    return result


def _row_features(row: pd.Series) -> dict[str, float]:
    return {col: float(row[col]) for col in FEATURE_COLUMNS if col in row and pd.notna(row[col])}


def _trade_pnl(trade: Trade) -> float:
    if trade.exit is None:
        return 0.0
    move = (trade.exit - trade.entry) if trade.direction == "BUY" else (trade.entry - trade.exit)
    return move * _pnl_per_price_unit(trade.lot)
