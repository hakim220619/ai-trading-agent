"""Daily trading limit checks for live order entry."""
from __future__ import annotations

from datetime import datetime, time, timezone
from typing import Any

from app.config import settings
from app.mt5.connection import MT5_AVAILABLE, connection, mt5
from app.runtime_config import get_trading_setup


def _today_window_utc() -> tuple[datetime, datetime]:
    now_local = datetime.now().astimezone()
    start_local = datetime.combine(now_local.date(), time.min, tzinfo=now_local.tzinfo)
    return start_local.astimezone(timezone.utc), now_local.astimezone(timezone.utc)


def daily_summary(bot_only: bool = True) -> dict[str, Any]:
    """Return today's realized/floating P/L and opened lot count."""
    if not MT5_AVAILABLE or not connection.ensure_connected():
        return {
            "profit": 0.0,
            "realized_profit": 0.0,
            "floating_profit": 0.0,
            "lot": 0.0,
            "start_utc": None,
            "end_utc": None,
        }
    start, end = _today_window_utc()
    deals = mt5.history_deals_get(start, end) or []
    realized = 0.0
    opened_lot = 0.0
    for deal in deals:
        if bot_only and deal.magic != settings.magic_number:
            continue
        if deal.entry in {mt5.DEAL_ENTRY_OUT, mt5.DEAL_ENTRY_OUT_BY}:
            realized += sum(float(getattr(deal, name, 0.0) or 0.0) for name in ("profit", "commission", "swap", "fee"))
        elif deal.entry == mt5.DEAL_ENTRY_IN:
            opened_lot += float(deal.volume or 0.0)
    positions = mt5.positions_get() or []
    floating = sum(
        float(position.profit or 0.0)
        for position in positions
        if not bot_only or position.magic == settings.magic_number
    )
    return {
        "profit": round(realized + floating, 2),
        "realized_profit": round(realized, 2),
        "floating_profit": round(floating, 2),
        "lot": round(opened_lot, 2),
        "start_utc": start.isoformat(),
        "end_utc": end.isoformat(),
    }


def check_daily_limits(next_lot: float = 0.0, bot_only: bool = True) -> tuple[bool, str, dict[str, Any]]:
    """Validate configured per-day profit and lot caps before a new order."""
    setup = get_trading_setup()
    summary = daily_summary(bot_only=bot_only)
    if bool(setup["daily_profit_limit_enabled"]):
        limit = float(setup["daily_profit_limit_money"])
        if limit > 0 and float(summary["profit"]) >= limit:
            return False, f"daily profit limit reached ({summary['profit']}/{limit})", summary
    if bool(setup["daily_lot_limit_enabled"]):
        limit = float(setup["daily_lot_limit"])
        projected = float(summary["lot"]) + max(0.0, float(next_lot))
        if limit > 0 and projected > limit:
            detail = {**summary, "projected_lot": round(projected, 2)}
            return False, f"daily lot limit reached ({projected:.2f}/{limit:.2f})", detail
    return True, "ok", summary
