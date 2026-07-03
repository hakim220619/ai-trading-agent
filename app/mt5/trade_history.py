"""Read closed trade results from the connected MT5 account."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from app.config import settings
from app.mt5.connection import MT5_AVAILABLE, connection, mt5
from app.mt5.confidence_metadata import parse_confidence_pct, parse_cycle_key


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
    entry_types = {mt5.DEAL_ENTRY_IN}
    entries: dict[int, Any] = {}
    for deal in deals:
        if deal.entry in entry_types and deal.position_id:
            previous = entries.get(deal.position_id)
            if previous is None or deal.time < previous.time:
                entries[deal.position_id] = deal
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
        opening = entries.get(deal.position_id)
        confidence_pct = parse_confidence_pct(getattr(opening, "comment", "")) if opening else None
        cycle_key = parse_cycle_key(getattr(opening, "comment", "")) if opening else None
        rows.append({
            "ticket": deal.ticket,
            "position_id": deal.position_id,
            "time": datetime.fromtimestamp(deal.time, tz=timezone.utc).isoformat(),
            "open_time": datetime.fromtimestamp(opening.time, tz=timezone.utc).isoformat() if opening else None,
            "symbol": deal.symbol,
            "type_str": ("BUY" if opening.type == mt5.DEAL_TYPE_BUY else "SELL") if opening else ("SELL" if deal.type == mt5.DEAL_TYPE_BUY else "BUY"),
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
            "confidence_pct": confidence_pct,
            "cycle_key": cycle_key,
        })
    rows.sort(key=lambda item: item["time"], reverse=True)
    return rows if limit is None else rows[: max(1, limit)]


def summarize_closed_deals(deals: list[dict[str, Any]]) -> dict[str, float | int]:
    wins = sum(float(d["net_profit"]) > 0 for d in deals)
    losses = sum(float(d["net_profit"]) < 0 for d in deals)
    profitable_confidences = [
        float(d["confidence_pct"])
        for d in deals
        if float(d["net_profit"]) > 0 and d.get("confidence_pct") is not None
    ]
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
        "average_profit_confidence_pct": round(sum(profitable_confidences) / len(profitable_confidences), 2) if profitable_confidences else None,
        "profit_confidence_trades": len(profitable_confidences),
    }


def summarize_by_open_hour(deals: list[dict[str, Any]], timezone_offset_minutes: int = 0) -> dict[str, Any]:
    """Aggregate realized results by local opening hour."""
    rows = {hour: {"hour": hour, "trades": 0, "wins": 0, "losses": 0, "gross_profit": 0.0, "gross_loss": 0.0, "net_profit": 0.0} for hour in range(24)}
    analyzed = 0
    for deal in deals:
        source_time = deal.get("open_time")
        if not source_time:
            continue
        opened = datetime.fromisoformat(str(source_time)) - timedelta(minutes=timezone_offset_minutes)
        row = rows[opened.hour]
        pnl = float(deal.get("net_profit", 0.0))
        row["trades"] += 1
        row["wins"] += int(pnl > 0)
        row["losses"] += int(pnl < 0)
        row["gross_profit"] += max(pnl, 0.0)
        row["gross_loss"] += min(pnl, 0.0)
        row["net_profit"] += pnl
        analyzed += 1
    active = []
    for row in rows.values():
        if not row["trades"]:
            continue
        decided = row["wins"] + row["losses"]
        row["win_rate_pct"] = round(row["wins"] / decided * 100, 2) if decided else 0.0
        for key in ("gross_profit", "gross_loss", "net_profit"):
            row[key] = round(row[key], 2)
        row["label"] = f"{row['hour']:02d}:00–{(row['hour'] + 1) % 24:02d}:00"
        active.append(row)
    ranked = sorted(active, key=lambda row: (row["net_profit"], row["win_rate_pct"], row["trades"]), reverse=True)
    recommended = [row for row in ranked if row["net_profit"] > 0][:3]
    return {
        "deals_analyzed": analyzed,
        "hours": sorted(active, key=lambda row: row["hour"]),
        "recommended_hours": recommended,
        "best_hour": recommended[0] if recommended else None,
        "timezone_offset_minutes": timezone_offset_minutes,
    }


def get_capital_curve(days: int = 3650, max_points: int = 240) -> dict[str, Any]:
    """Reconstruct the active account balance and trading-only growth."""
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
        net_deposits = initial_balance
    else:
        initial_balance = current_balance - sum(change for _, change, _ in changes)
        start_time, remaining = start, changes
        net_deposits = initial_balance
    balance = initial_balance
    points: list[dict[str, Any]] = [{"time": start_time.isoformat(), "balance": round(balance, 2), "growth": 0.0}]
    for timestamp, change, is_balance in remaining:
        balance += change
        if is_balance:
            net_deposits += change
        points.append({
            "time": timestamp.isoformat(),
            "balance": round(balance, 2),
            "growth": round(balance - net_deposits, 2),
        })
    trading_growth = current_balance - net_deposits
    points.append({
        "time": end.isoformat(),
        "balance": round(current_balance, 2),
        "equity": round(current_equity, 2),
        "growth": round(trading_growth, 2),
    })
    if len(points) > max_points:
        step = max(1, len(points) // (max_points - 1))
        points = points[::step]
        if points[-1].get("time") != end.isoformat():
            points.append({"time": end.isoformat(), "balance": round(current_balance, 2), "equity": round(current_equity, 2), "growth": round(trading_growth, 2)})
    return {
        "account_login": account.get("login"),
        "account_server": account.get("server"),
        "initial_balance": round(initial_balance, 2),
        "net_deposits": round(net_deposits, 2),
        "trading_growth": round(trading_growth, 2),
        "growth_pct": round(trading_growth / net_deposits * 100, 2) if net_deposits else 0.0,
        "current_balance": round(current_balance, 2),
        "current_equity": round(current_equity, 2),
        "currency": account.get("currency"),
        "points": points,
    }
