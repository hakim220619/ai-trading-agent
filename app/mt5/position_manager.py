"""High-level position management: profit targets, trailing, reversal exits."""
from __future__ import annotations

from typing import Any

from app.config import settings
from app.mt5.connection import MT5_AVAILABLE, connection, mt5
from app.mt5 import order_executor
from app.utils.logger import logger


def get_open_positions(symbol: str | None = None) -> list[dict[str, Any]]:
    """Return bot-owned open positions as dicts."""
    if not MT5_AVAILABLE or not connection.ensure_connected():
        return []
    positions = mt5.positions_get(symbol=symbol or settings.symbol) or []
    out = []
    for p in positions:
        if p.magic != settings.magic_number:
            continue
        d = p._asdict()
        d["type_str"] = "BUY" if p.type == mt5.POSITION_TYPE_BUY else "SELL"
        out.append(d)
    return out


def total_profit(symbol: str | None = None) -> float:
    """Sum of floating profit across bot-owned positions."""
    return sum(p["profit"] for p in get_open_positions(symbol))


def manage_positions(symbol: str | None = None, new_signal: str | None = None) -> dict[str, Any]:
    """Run the per-tick position management routine.

    Order of operations:
      1. If total floating profit >= target, close everything.
      2. Update trailing stops.
      3. If a fresh opposite signal arrived, close conflicting positions.
    """
    symbol = symbol or settings.symbol
    actions: dict[str, Any] = {"closed_for_target": [], "trailing": [], "closed_for_reversal": []}

    profit = total_profit(symbol)
    if profit >= settings.target_profit_money and self_has_positions(symbol):
        logger.success("Profit target hit ({} >= {}). Closing all.", profit, settings.target_profit_money)
        actions["closed_for_target"] = order_executor.close_all_positions(symbol)
        return actions

    actions["trailing"] = order_executor.update_trailing_stop(symbol)

    if new_signal in ("BUY", "SELL"):
        opposite = "SELL" if new_signal == "BUY" else "BUY"
        if order_executor.has_open_position(symbol, opposite.lower()):
            logger.info("Opposite signal {} - closing existing {} positions.", new_signal, opposite)
            actions["closed_for_reversal"] = _close_direction(symbol, opposite)

    return actions


def self_has_positions(symbol: str | None = None) -> bool:
    return len(get_open_positions(symbol)) > 0


def _close_direction(symbol: str, direction: str) -> list[dict[str, Any]]:
    """Close only positions matching a given direction (BUY/SELL)."""
    if not MT5_AVAILABLE or not connection.ensure_connected():
        return []
    want = mt5.POSITION_TYPE_BUY if direction == "BUY" else mt5.POSITION_TYPE_SELL
    positions = mt5.positions_get(symbol=symbol) or []
    results = []
    for p in positions:
        if p.magic != settings.magic_number or p.type != want:
            continue
        res = order_executor._close_position(p)  # internal helper reuse
        results.append({"ticket": p.ticket, **res.to_dict()})
    return results
