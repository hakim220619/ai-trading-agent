"""Read closed trade results from the connected MT5 account."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from app.config import settings
from app.mt5.connection import MT5_AVAILABLE, connection, mt5


def get_closed_deals(
    days: int = 30,
    limit: int | None = 100,
    symbol: str | None = None,
    bot_only: bool = False,
) -> list[dict[str, Any]]:
    """Return closing deals, newest first, including their net realized P/L."""
    if not MT5_AVAILABLE or not connection.ensure_connected():
        return []
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=max(1, days))
    deals = mt5.history_deals_get(start, end) or []
    exit_types = {mt5.DEAL_ENTRY_OUT, mt5.DEAL_ENTRY_OUT_BY}
    rows: list[dict[str, Any]] = []
    for deal in deals:
        if deal.entry not in exit_types:
            continue
        if symbol and deal.symbol != symbol:
            continue
        if bot_only and deal.magic != settings.magic_number:
            continue
        commission = float(deal.commission or 0.0)
        swap = float(deal.swap or 0.0)
        fee = float(getattr(deal, "fee", 0.0) or 0.0)
        profit = float(deal.profit or 0.0)
        net_profit = profit + commission + swap + fee
        rows.append({
            "ticket": deal.ticket,
            "position_id": deal.position_id,
            "time": datetime.fromtimestamp(deal.time, tz=timezone.utc).isoformat(),
            "symbol": deal.symbol,
            "type_str": "BUY" if deal.type == mt5.DEAL_TYPE_BUY else "SELL",
            "volume": float(deal.volume),
            "price": float(deal.price),
            "profit": profit,
            "commission": commission,
            "swap": swap,
            "fee": fee,
            "net_profit": net_profit,
            "result": "WIN" if net_profit > 0 else "LOSS" if net_profit < 0 else "BE",
            "magic": deal.magic,
            "comment": deal.comment,
        })
    rows.sort(key=lambda item: item["time"], reverse=True)
    return rows if limit is None else rows[: max(1, limit)]


def summarize_closed_deals(deals: list[dict[str, Any]]) -> dict[str, float | int]:
    wins = sum(float(d["net_profit"]) > 0 for d in deals)
    losses = sum(float(d["net_profit"]) < 0 for d in deals)
    return {
        "count": len(deals),
        "wins": wins,
        "losses": losses,
        "breakeven": len(deals) - wins - losses,
        "win_rate_pct": round(wins / (wins + losses) * 100, 2) if wins + losses else 0.0,
        "gross_profit": round(sum(max(float(d["net_profit"]), 0.0) for d in deals), 2),
        "gross_loss": round(sum(min(float(d["net_profit"]), 0.0) for d in deals), 2),
        "net_profit": round(sum(float(d["net_profit"]) for d in deals), 2),
        "commission": round(sum(float(d["commission"]) for d in deals), 2),
        "swap": round(sum(float(d["swap"]) for d in deals), 2),
    }


def get_capital_curve(days: int = 3650, max_points: int = 240) -> dict[str, Any]:
    """Reconstruct realized account balance and append current equity."""
    if not MT5_AVAILABLE or not connection.ensure_connected():
        return {"initial_balance": None, "current_balance": None, "current_equity": None, "points": []}
    account = connection.account_info() or {}
    current_balance = float(account.get("balance", 0.0) or 0.0)
    current_equity = float(account.get("equity", current_balance) or current_balance)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=max(1, days))
    raw_deals = mt5.history_deals_get(start, end) or []
    changes: list[tuple[datetime, float, bool]] = []
    for deal in raw_deals:
        change = sum(float(getattr(deal, name, 0.0) or 0.0) for name in ("profit", "commission", "swap", "fee"))
        if change:
            changes.append((
                datetime.fromtimestamp(deal.time, tz=timezone.utc),
                change,
                deal.type == mt5.DEAL_TYPE_BALANCE,
            ))
    changes.sort(key=lambda item: item[0])
    first_deposit = next(((timestamp, change) for timestamp, change, is_balance in changes if is_balance and change > 0), None)
    if first_deposit:
        start_time, initial_balance = first_deposit
        remaining = [item for item in changes if item[0] > start_time]
    else:
        initial_balance = current_balance - sum(change for _, change, _ in changes)
        start_time, remaining = start, changes
    balance = initial_balance
    points: list[dict[str, Any]] = [{"time": start_time.isoformat(), "balance": round(balance, 2)}]
    for timestamp, change, _ in remaining:
        balance += change
        points.append({"time": timestamp.isoformat(), "balance": round(balance, 2)})
    points.append({"time": end.isoformat(), "balance": round(current_balance, 2), "equity": round(current_equity, 2)})
    if len(points) > max_points:
        step = max(1, len(points) // (max_points - 1))
        points = points[::step]
        if points[-1].get("time") != end.isoformat():
            points.append({"time": end.isoformat(), "balance": round(current_balance, 2), "equity": round(current_equity, 2)})
    return {
        "initial_balance": round(initial_balance, 2),
        "current_balance": round(current_balance, 2),
        "current_equity": round(current_equity, 2),
        "currency": account.get("currency"),
        "points": points,
    }
