"""Intraday backtester for the standalone universal day-trade strategy."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from app.config import settings
from app.mt5.connection import connection
from app.strategy.day_trade_strategy import (
    build_day_trade_plan,
    generate_day_trade_signal,
    get_day_trade_preset,
    prepare_day_trade_features,
)


@dataclass
class DayTrade:
    direction: str
    day: str
    entry_time: object
    entry_idx: int
    entry: float
    sl: float
    tp: float
    initial_risk: float
    lot: float
    pnl_per_unit: float
    spread_cost: float = 0.0
    commission: float = 0.0
    slippage_cost: float = 0.0
    exit_time: object | None = None
    exit: float | None = None
    pnl: float = 0.0
    gross_pnl: float = 0.0
    result: str = "open"


@dataclass
class DayTradeResult:
    trades: list[DayTrade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    daily_pnl: dict[str, float] = field(default_factory=dict)
    start_balance: float = 0.0
    end_balance: float = 0.0
    daily_profit_target: float = 30.0
    daily_loss_limit: float = 20.0
    account_profile: str = "exness-pro"
    market_profile: str = "standard"


def _contract() -> tuple[float, float, float]:
    """Return point, tick size and tick value for the configured symbol."""
    info = connection.symbol_info(settings.symbol) or {}
    point = float(info.get("point", 0.01) or 0.01)
    tick_size = float(info.get("trade_tick_size", point) or point)
    tick_value = float(info.get("trade_tick_value", 1.0) or 1.0)
    return point, tick_size, tick_value


def run_day_trade_backtest(
    df: pd.DataFrame,
    start_balance: float = 100.0,
    daily_profit_target: float = 30.0,
    daily_loss_limit: float = 20.0,
    risk_percent: float = 1.0,
    max_trades_per_day: int = 3,
    max_hold_bars: int = 24,
    cooldown_bars: int = 6,
    max_spread_points: float = 1500.0,
    commission_per_lot_side: float = 0.0,
    slippage_points: float = 0.0,
    account_profile: str = "exness-pro",
    market_profile: str = "auto",
) -> DayTradeResult:
    """Replay a multi-market day strategy with daily profit/loss circuit breakers."""
    if df is None or len(df) < 250:
        raise ValueError("At least 250 candles are required for day-trade backtest")
    preset = get_day_trade_preset(settings.symbol, market_profile)
    frame = prepare_day_trade_features(df, breakout_period=preset.breakout_period).reset_index(drop=True)
    if "time" not in frame:
        raise ValueError("CSV requires a time column for daily limits")
    frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
    frame = frame.dropna(subset=["time"]).reset_index(drop=True)

    point, tick_size, tick_value = _contract()
    result = DayTradeResult(
        start_balance=start_balance,
        daily_profit_target=daily_profit_target,
        daily_loss_limit=daily_loss_limit,
        account_profile=account_profile,
        market_profile=preset.name,
    )
    balance = start_balance
    open_trade: DayTrade | None = None
    current_day = ""
    trades_today = 0
    last_exit_idx = -10_000

    def spread_distance(row: pd.Series) -> float:
        value = float(row.get("spread", 0.0) or 0.0)
        return max(value, 0.0) * point

    def finalize(trade: DayTrade, exit_price: float, exit_time: object, exit_spread: float) -> None:
        nonlocal balance, open_trade, last_exit_idx
        trade.exit = exit_price
        trade.exit_time = exit_time
        if trade.direction == "SELL":
            trade.spread_cost = exit_spread * trade.pnl_per_unit
        move = exit_price - trade.entry if trade.direction == "BUY" else trade.entry - exit_price
        trade.pnl = move * trade.pnl_per_unit - trade.commission
        trade.gross_pnl = trade.pnl + trade.spread_cost + trade.commission + trade.slippage_cost
        trade.result = "win" if trade.pnl > 0 else "loss"
        balance += trade.pnl
        result.daily_pnl[trade.day] = result.daily_pnl.get(trade.day, 0.0) + trade.pnl
        result.trades.append(trade)
        last_exit_idx = i
        open_trade = None

    for i in range(200, len(frame)):
        row = frame.iloc[i]
        day = row["time"].date().isoformat()
        result.daily_pnl.setdefault(day, 0.0)
        slip = max(slippage_points, 0.0) * point

        # Day trades never remain open across the UTC date boundary.
        if current_day and day != current_day:
            if open_trade is not None:
                previous = frame.iloc[i - 1]
                previous_spread = spread_distance(previous)
                exit_price = (
                    float(previous["close"]) - slip
                    if open_trade.direction == "BUY"
                    else float(previous["close"]) + previous_spread + slip
                )
                finalize(open_trade, exit_price, previous["time"], previous_spread)
            trades_today = 0
        current_day = day

        current_spread = spread_distance(row)
        spread_points = current_spread / point if point else 0.0

        if open_trade is not None:
            high, low = float(row["high"]), float(row["low"])
            exit_price: float | None = None
            if open_trade.direction == "BUY":
                if low <= open_trade.sl:
                    exit_price = open_trade.sl - slip
                elif high >= open_trade.tp:
                    exit_price = open_trade.tp - slip
                elif high >= open_trade.entry + open_trade.initial_risk:
                    open_trade.sl = max(open_trade.sl, open_trade.entry)
            else:
                ask_high, ask_low = high + current_spread, low + current_spread
                if ask_high >= open_trade.sl:
                    exit_price = open_trade.sl + slip
                elif ask_low <= open_trade.tp:
                    exit_price = open_trade.tp + slip
                elif ask_low <= open_trade.entry - open_trade.initial_risk:
                    open_trade.sl = min(open_trade.sl, open_trade.entry)
            if exit_price is None and i - open_trade.entry_idx >= max_hold_bars:
                exit_price = (
                    float(row["close"]) - slip
                    if open_trade.direction == "BUY"
                    else float(row["close"]) + current_spread + slip
                )
            if exit_price is not None:
                finalize(open_trade, exit_price, row["time"], current_spread)

        result.equity_curve.append(balance)
        daily_realized = result.daily_pnl.get(day, 0.0)
        blocked_for_day = (
            daily_realized >= daily_profit_target
            or daily_realized <= -daily_loss_limit
            or trades_today >= max_trades_per_day
        )
        if open_trade is None and not blocked_for_day and i - last_exit_idx >= cooldown_bars:
            signal = generate_day_trade_signal(
                row,
                spread_points,
                max_spread_points,
                preset=preset,
            )
            if signal.action in ("BUY", "SELL"):
                bid = float(row["close"])
                entry = bid + current_spread + slip if signal.action == "BUY" else bid - slip
                plan = build_day_trade_plan(
                    signal.action,
                    entry,
                    signal.atr,
                    balance,
                    settings.symbol,
                    risk_percent=risk_percent,
                    stop_atr=preset.stop_atr,
                    risk_reward=preset.risk_reward,
                )
                pnl_per_unit = (tick_value / tick_size) * plan.lot
                open_trade = DayTrade(
                    direction=signal.action,
                    day=day,
                    entry_time=row["time"],
                    entry_idx=i,
                    entry=plan.entry,
                    sl=plan.stop_loss,
                    tp=plan.take_profit,
                    initial_risk=abs(plan.entry - plan.stop_loss),
                    lot=plan.lot,
                    pnl_per_unit=pnl_per_unit,
                    spread_cost=current_spread * pnl_per_unit if signal.action == "BUY" else 0.0,
                    commission=max(commission_per_lot_side, 0.0) * plan.lot * 2.0,
                    slippage_cost=slip * pnl_per_unit * 2.0,
                )
                trades_today += 1

    if open_trade is not None:
        row = frame.iloc[-1]
        final_spread = spread_distance(row)
        slip = max(slippage_points, 0.0) * point
        exit_price = (
            float(row["close"]) - slip
            if open_trade.direction == "BUY"
            else float(row["close"]) + final_spread + slip
        )
        finalize(open_trade, exit_price, row["time"], final_spread)
    result.end_balance = balance
    return result


def summarize_day_trade(result: DayTradeResult) -> dict[str, Any]:
    trades = result.trades
    wins = [trade for trade in trades if trade.pnl > 0]
    losses = [trade for trade in trades if trade.pnl <= 0]
    gross_profit = sum(trade.pnl for trade in wins)
    gross_loss = abs(sum(trade.pnl for trade in losses))
    net = result.end_balance - result.start_balance
    spread_cost = sum(trade.spread_cost for trade in trades)
    commission = sum(trade.commission for trade in trades)
    slippage = sum(trade.slippage_cost for trade in trades)
    active_days = [value for value in result.daily_pnl.values() if value != 0]
    profitable_days = [value for value in active_days if value > 0]
    losing_days = [value for value in active_days if value < 0]
    return {
        "strategy": "universal-intraday-momentum-breakout",
        "account_profile": result.account_profile,
        "market_profile": result.market_profile,
        "trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": round(len(wins) / len(trades) * 100, 2) if trades else 0.0,
        "net_profit": round(net, 2),
        "roi_pct": round(net / result.start_balance * 100, 2) if result.start_balance else 0.0,
        "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss else "inf",
        "max_drawdown_pct": round(_max_drawdown(result.equity_curve), 2),
        "spread_cost": round(spread_cost, 2),
        "commission_cost": round(commission, 2),
        "slippage_cost": round(slippage, 2),
        "days_tested": len(result.daily_pnl),
        "active_trading_days": len(active_days),
        "profitable_days": len(profitable_days),
        "losing_days": len(losing_days),
        "daily_target": result.daily_profit_target,
        "target_hit_days": sum(value >= result.daily_profit_target for value in active_days),
        "average_daily_pnl": round(sum(active_days) / len(active_days), 2) if active_days else 0.0,
        "best_day": round(max(active_days), 2) if active_days else 0.0,
        "worst_day": round(min(active_days), 2) if active_days else 0.0,
        "start_balance": round(result.start_balance, 2),
        "end_balance": round(result.end_balance, 2),
    }


def _max_drawdown(equity: list[float]) -> float:
    if not equity:
        return 0.0
    peak, maximum = equity[0], 0.0
    for value in equity:
        peak = max(peak, value)
        if peak > 0:
            maximum = max(maximum, (peak - value) / peak * 100.0)
    return maximum
