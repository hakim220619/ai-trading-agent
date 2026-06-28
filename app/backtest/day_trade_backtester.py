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
    strategy: str = "day"
    fixed_lot: float | None = None
    session: str = "24h"


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
    strategy: str = "day",
    fixed_lot: float | None = None,
    weekdays_only: bool = False,
    session_start_hour: int | None = None,
    session_hours: int = 24,
    m1_min_setups: int = 3,
    max_consecutive_losses: int = 0,
) -> DayTradeResult:
    """Replay a multi-market day strategy with daily profit/loss circuit breakers."""
    if df is None or len(df) < 250:
        raise ValueError("At least 250 candles are required for day-trade backtest")
    if fixed_lot is not None and fixed_lot <= 0:
        raise ValueError("fixed_lot must be greater than zero")
    if session_start_hour is not None and not 0 <= session_start_hour <= 23:
        raise ValueError("session_start_hour must be between 0 and 23")
    if not 1 <= session_hours <= 24:
        raise ValueError("session_hours must be between 1 and 24")
    strategy = strategy.lower()
    if strategy == "ma-m1":
        from app.strategy.scalping_ma_m1_strategy import prepare_ma_m1_features

        preset = None
        frame = prepare_ma_m1_features(df).reset_index(drop=True)
        profile_name = f"{settings.symbol}-ma-m1"
    elif strategy == "snd-m1":
        from app.strategy.scalping_snd_m1_strategy import prepare_snd_m1_features

        preset = None
        frame = prepare_snd_m1_features(df).reset_index(drop=True)
        profile_name = f"{settings.symbol}-snd-m1"
    elif strategy == "m1":
        from app.strategy.scalping_m1_strategy import get_m1_preset, prepare_m1_features

        preset = get_m1_preset(settings.symbol)
        frame = prepare_m1_features(df).reset_index(drop=True)
        profile_name = f"{settings.symbol}-m1"
    elif strategy == "m5":
        from app.strategy.scalping_m5_strategy import get_m5_preset, prepare_m5_features

        preset = get_m5_preset(settings.symbol)
        frame = prepare_m5_features(df, settings.symbol).reset_index(drop=True)
        profile_name = f"{settings.symbol}-m5"
    elif strategy == "day":
        preset = get_day_trade_preset(settings.symbol, market_profile)
        frame = prepare_day_trade_features(df, breakout_period=preset.breakout_period).reset_index(drop=True)
        profile_name = preset.name
    else:
        raise ValueError("strategy must be day, m1, ma-m1, snd-m1, or m5")
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
        market_profile=profile_name,
        strategy=strategy,
        fixed_lot=fixed_lot,
        session=(
            f"{session_start_hour:02d}:00-{(session_start_hour + session_hours) % 24:02d}:00"
            if session_start_hour is not None else "24h"
        ),
    )
    balance = start_balance
    open_trade: DayTrade | None = None
    current_day = ""
    trades_today = 0
    consecutive_losses = 0
    last_exit_idx = -10_000

    def spread_distance(row: pd.Series) -> float:
        value = float(row.get("spread", 0.0) or 0.0)
        return max(value, 0.0) * point

    def finalize(trade: DayTrade, exit_price: float, exit_time: object, exit_spread: float) -> None:
        nonlocal balance, open_trade, last_exit_idx, consecutive_losses
        trade.exit = exit_price
        trade.exit_time = exit_time
        if trade.direction == "SELL":
            trade.spread_cost = exit_spread * trade.pnl_per_unit
        move = exit_price - trade.entry if trade.direction == "BUY" else trade.entry - exit_price
        trade.pnl = move * trade.pnl_per_unit - trade.commission
        trade.gross_pnl = trade.pnl + trade.spread_cost + trade.commission + trade.slippage_cost
        trade.result = "win" if trade.pnl > 0 else "loss"
        consecutive_losses = 0 if trade.pnl > 0 else consecutive_losses + 1
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
            consecutive_losses = 0
        current_day = day

        current_spread = spread_distance(row)
        spread_points = current_spread / point if point else 0.0
        timestamp = row["time"]
        weekday_ok = not weekdays_only or timestamp.weekday() < 5
        if session_start_hour is None:
            session_ok = True
        else:
            minute_of_day = timestamp.hour * 60 + timestamp.minute
            start_minute = session_start_hour * 60
            elapsed = (minute_of_day - start_minute) % (24 * 60)
            session_ok = elapsed < session_hours * 60

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
            # The bot is off outside its configured daily running window.
            if exit_price is None and (not weekday_ok or not session_ok):
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
            or (max_consecutive_losses > 0 and consecutive_losses >= max_consecutive_losses)
        )
        if open_trade is None and weekday_ok and session_ok and not blocked_for_day and i - last_exit_idx >= cooldown_bars:
            if strategy == "ma-m1":
                from app.strategy.scalping_ma_m1_strategy import generate_ma_m1_signal

                signal = generate_ma_m1_signal(row, settings.symbol, spread_points, max_spread_points)
            elif strategy == "snd-m1":
                from app.strategy.scalping_snd_m1_strategy import generate_snd_m1_signal

                signal = generate_snd_m1_signal(row, settings.symbol, spread_points, max_spread_points)
            elif strategy == "m1":
                from app.strategy.scalping_m1_strategy import generate_m1_signal

                signal = generate_m1_signal(
                    row, settings.symbol, spread_points, max_spread_points,
                    min_setup_confirmations=m1_min_setups,
                )
            elif strategy == "m5":
                from app.strategy.scalping_m5_strategy import generate_m5_signal

                signal = generate_m5_signal(row, settings.symbol, spread_points, max_spread_points)
            else:
                signal = generate_day_trade_signal(row, spread_points, max_spread_points, preset=preset)
            if signal.action in ("BUY", "SELL"):
                bid = float(row["close"])
                entry = bid + current_spread + slip if signal.action == "BUY" else bid - slip
                if strategy == "ma-m1":
                    from app.strategy.scalping_ma_m1_strategy import build_ma_m1_plan

                    plan = build_ma_m1_plan(signal.action, entry, signal.atr, balance, settings.symbol, risk_percent)
                elif strategy == "snd-m1":
                    from app.strategy.scalping_snd_m1_strategy import build_snd_m1_plan

                    plan = build_snd_m1_plan(signal.action, entry, signal.atr, balance, settings.symbol, risk_percent)
                elif strategy == "m1":
                    from app.strategy.scalping_m1_strategy import build_m1_plan

                    plan = build_m1_plan(signal.action, entry, signal.atr, balance, settings.symbol, risk_percent)
                elif strategy == "m5":
                    from app.strategy.scalping_m5_strategy import build_m5_plan

                    plan = build_m5_plan(signal.action, entry, signal.atr, balance, settings.symbol, risk_percent)
                else:
                    plan = build_day_trade_plan(
                        signal.action, entry, signal.atr, balance, settings.symbol,
                        risk_percent=risk_percent, stop_atr=preset.stop_atr,
                        risk_reward=preset.risk_reward,
                    )
                lot = fixed_lot if fixed_lot is not None else plan.lot
                pnl_per_unit = (tick_value / tick_size) * lot
                open_trade = DayTrade(
                    direction=signal.action,
                    day=day,
                    entry_time=row["time"],
                    entry_idx=i,
                    entry=plan.entry,
                    sl=plan.stop_loss,
                    tp=plan.take_profit,
                    initial_risk=abs(plan.entry - plan.stop_loss),
                    lot=lot,
                    pnl_per_unit=pnl_per_unit,
                    spread_cost=current_spread * pnl_per_unit if signal.action == "BUY" else 0.0,
                    commission=max(commission_per_lot_side, 0.0) * lot * 2.0,
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
        "strategy": (
            "universal-intraday-momentum-breakout"
            if result.strategy == "day"
            else f"scalping-{result.strategy}"
        ),
        "account_profile": result.account_profile,
        "market_profile": result.market_profile,
        "position_sizing": f"fixed {result.fixed_lot:.2f} lot" if result.fixed_lot is not None else "risk based",
        "trading_session": result.session,
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
