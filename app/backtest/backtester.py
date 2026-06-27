"""Event-driven backtester over historical candles.

Replays candles bar-by-bar, generating signals with the same rule+ML engine the
live bot uses, then simulates SL/TP fills on subsequent bars.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from app.config import settings
from app.ml.feature_engineering import build_features, extract_feature_row
from app.mt5.connection import connection
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
    gross_pnl: float = 0.0
    spread_cost: float = 0.0
    commission: float = 0.0
    slippage_cost: float = 0.0
    pnl_per_price_unit: float = 0.0
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
            "gross_pnl": round(self.gross_pnl, 2),
            "spread_cost": round(self.spread_cost, 2),
            "commission": round(self.commission, 2),
            "slippage_cost": round(self.slippage_cost, 2),
            "result": self.result,
        }


@dataclass
class BacktestResult:
    """Aggregate backtest output."""

    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    start_balance: float = 0.0
    end_balance: float = 0.0
    account_profile: str = "custom"
    historical_spread: bool = True
    commission_per_lot_side: float = 0.0
    slippage_points: float = 0.0


def _pnl_per_price_unit(lot: float) -> float:
    """PnL per 1.0 price move using the broker's active symbol contract."""
    info = connection.symbol_info(settings.symbol)
    if info:
        tick_size = float(info.get("trade_tick_size", 0.0) or 0.0)
        tick_value = float(info.get("trade_tick_value", 0.0) or 0.0)
        if tick_size > 0 and tick_value > 0:
            return (tick_value / tick_size) * lot
    # Offline fallback matches the default XAUUSD contract approximation.
    return 100.0 * lot


def run_backtest(
    df: pd.DataFrame,
    start_balance: float = 1000.0,
    warmup: int = 200,
    max_hold: int = 96,
    signal_lookback: int = 500,
    account_profile: str = "exness-pro",
    use_historical_spread: bool = True,
    commission_per_lot_side: float = 0.0,
    slippage_points: float = 0.0,
) -> BacktestResult:
    """Run a vectorised-ish backtest.

    ``warmup`` bars are skipped so indicators are valid. Each open trade is held
    until SL/TP hits or ``max_hold`` bars elapse. Only one position at a time.
    """
    if df is None or len(df) <= warmup + 10:
        raise ValueError("Not enough data to backtest.")

    df = build_features(df).reset_index(drop=True)
    result = BacktestResult(
        start_balance=start_balance,
        account_profile=account_profile,
        historical_spread=use_historical_spread,
        commission_per_lot_side=commission_per_lot_side,
        slippage_points=slippage_points,
    )
    symbol_info = connection.symbol_info(settings.symbol) or {}
    point = float(symbol_info.get("point", 0.01) or 0.01)
    balance = start_balance
    open_trade: Trade | None = None
    open_idx = 0

    for i in range(warmup, len(df) - 1):
        # A bounded history is both more realistic for current S/R levels and
        # keeps multi-year backtests from degrading to quadratic runtime.
        window_start = max(0, i - signal_lookback + 1)
        window = df.iloc[window_start : i + 1]
        row = df.iloc[i]
        spread_points = max(float(row.get("spread", 0.0) or 0.0), 0.0)
        spread_distance = spread_points * point if use_historical_spread else 0.0
        slippage_distance = max(slippage_points, 0.0) * point

        # --- manage open trade ---
        if open_trade is not None:
            high = float(row["high"])
            low = float(row["low"])
            closed = False
            if open_trade.direction == "BUY":
                if low <= open_trade.sl:
                    open_trade.exit = open_trade.sl - slippage_distance
                    closed = True
                elif high >= open_trade.tp:
                    open_trade.exit = open_trade.tp - slippage_distance
                    closed = True
            else:  # SELL
                # MT5 OHLC candles are bid prices. A SELL closes at ask, so its
                # SL/TP trigger must be checked against bid + historical spread.
                ask_high = high + spread_distance
                ask_low = low + spread_distance
                if ask_high >= open_trade.sl:
                    open_trade.exit = open_trade.sl + slippage_distance
                    closed = True
                elif ask_low <= open_trade.tp:
                    open_trade.exit = open_trade.tp + slippage_distance
                    closed = True

            if not closed and (i - open_idx) >= max_hold:
                bid_close = float(row["close"])
                open_trade.exit = (
                    bid_close - slippage_distance
                    if open_trade.direction == "BUY"
                    else bid_close + spread_distance + slippage_distance
                )
                closed = True

            if closed:
                open_trade.exit_time = row["time"] if "time" in row else i
                if open_trade.direction == "SELL":
                    open_trade.spread_cost = spread_distance * open_trade.pnl_per_price_unit
                open_trade.pnl = _trade_pnl(open_trade)
                open_trade.gross_pnl = (
                    open_trade.pnl
                    + open_trade.spread_cost
                    + open_trade.commission
                    + open_trade.slippage_cost
                )
                open_trade.result = "win" if open_trade.pnl > 0 else "loss"
                balance += open_trade.pnl
                result.trades.append(open_trade)
                open_trade = None

        result.equity_curve.append(balance)

        # --- look for new entry only when flat ---
        if open_trade is None:
            sig = generate_signal(
                window,
                features=extract_feature_row(row),
                spread_points=spread_points if use_historical_spread else 0.0,
            )
            if sig.action in ("BUY", "SELL"):
                bid_entry = float(row["close"])
                execution_entry = (
                    bid_entry + spread_distance + slippage_distance
                    if sig.action == "BUY"
                    else bid_entry - slippage_distance
                )
                plan = build_trade_plan(
                    direction=sig.action,
                    entry=execution_entry,
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
                open_trade.pnl_per_price_unit = _pnl_per_price_unit(plan.lot)
                open_trade.commission = (
                    max(commission_per_lot_side, 0.0) * plan.lot * 2.0
                )
                open_trade.slippage_cost = (
                    slippage_distance * open_trade.pnl_per_price_unit * 2.0
                )
                if sig.action == "BUY":
                    open_trade.spread_cost = (
                        spread_distance * open_trade.pnl_per_price_unit
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


def _trade_pnl(trade: Trade) -> float:
    if trade.exit is None:
        return 0.0
    move = (trade.exit - trade.entry) if trade.direction == "BUY" else (trade.entry - trade.exit)
    pnl_per_unit = trade.pnl_per_price_unit or _pnl_per_price_unit(trade.lot)
    return move * pnl_per_unit - trade.commission
