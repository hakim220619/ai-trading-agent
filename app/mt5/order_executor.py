"""Order execution against MetaTrader 5 with full pre-trade validation."""
from __future__ import annotations

from typing import Any

from app.config import settings
from app.mt5.connection import MT5_AVAILABLE, connection, mt5
from app.utils.helpers import round_to_step
from app.utils.logger import logger


class OrderResult:
    """Lightweight result wrapper for an order attempt."""

    def __init__(self, ok: bool, message: str, raw: Any = None) -> None:
        self.ok = ok
        self.message = message
        self.raw = raw

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "message": self.message}


def _validate_market_open(symbol: str) -> tuple[bool, str]:
    info = connection.symbol_info(symbol)
    if info is None:
        return False, "symbol info unavailable"
    # trade_mode 0 = disabled
    if info.get("trade_mode", 0) == 0:
        return False, "trading disabled for symbol"
    return True, "ok"


def _validate_spread(symbol: str) -> tuple[bool, str]:
    spread = connection.get_spread_points(symbol)
    if spread is None:
        return False, "spread unavailable"
    if spread > settings.max_spread_points:
        return False, f"spread {spread} > max {settings.max_spread_points}"
    return True, "ok"


def _validate_trade_permissions() -> tuple[bool, str]:
    status = connection.trading_status()
    if status["trade_api_disabled"]:
        return False, "external Python trading API is disabled in MT5"
    if not status["terminal_trade_allowed"]:
        return False, "Algo Trading is disabled in the MT5 terminal"
    if not status["account_trade_allowed"]:
        return False, "trading is not allowed for the logged-in account"
    return True, "ok"


def _normalize_lot(symbol: str, lot: float) -> float:
    info = connection.symbol_info(symbol)
    if info is None:
        return lot
    vol_min = info.get("volume_min", 0.01)
    vol_max = info.get("volume_max", 100.0)
    vol_step = info.get("volume_step", 0.01)
    lot = round_to_step(lot, vol_step)
    lot = max(vol_min, min(vol_max, lot))
    return round(lot, 2)


def _filling_type(symbol: str) -> int:
    """Choose a volume filling policy supported by the broker's symbol."""
    info = connection.symbol_info(symbol) or {}
    allowed = int(info.get("filling_mode", 0) or 0)
    if allowed & 2:  # SYMBOL_FILLING_IOC
        return mt5.ORDER_FILLING_IOC
    if allowed & 1:  # SYMBOL_FILLING_FOK
        return mt5.ORDER_FILLING_FOK
    execution = info.get("trade_exemode")
    if execution != mt5.SYMBOL_TRADE_EXECUTION_MARKET:
        return mt5.ORDER_FILLING_RETURN
    return mt5.ORDER_FILLING_FOK


def has_open_position(symbol: str, direction: str | None = None) -> bool:
    """True if there is already an open position (optionally filtered by side)."""
    if not MT5_AVAILABLE or not connection.ensure_connected():
        return False
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return False
    if direction is None:
        return True
    want = mt5.POSITION_TYPE_BUY if direction.lower() == "buy" else mt5.POSITION_TYPE_SELL
    return any(p.type == want and p.magic == settings.magic_number for p in positions)


def count_open_positions(symbol: str | None = None) -> int:
    """Count open positions owned by this bot (matching magic number)."""
    if not MT5_AVAILABLE or not connection.ensure_connected():
        return 0
    positions = mt5.positions_get(symbol=symbol or settings.symbol) or []
    return sum(1 for p in positions if p.magic == settings.magic_number)


def _send(request: dict[str, Any]) -> OrderResult:
    if request.get("action") == mt5.TRADE_ACTION_DEAL:
        check = mt5.order_check(request)
        if check is None:
            return OrderResult(False, f"order_check None: {mt5.last_error()}")
        if check.retcode != 0:
            return OrderResult(False, f"order_check retcode={check.retcode} {check.comment}", check)
    result = mt5.order_send(request)
    if result is None:
        return OrderResult(False, f"order_send None: {mt5.last_error()}")
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        return OrderResult(False, f"retcode={result.retcode} {result.comment}", result)
    return OrderResult(True, f"deal={result.deal} order={result.order}", result)


def _open(
    symbol: str,
    direction: str,
    lot: float,
    sl: float,
    tp: float,
    comment: str,
) -> OrderResult:
    """Shared open-position routine with all safety checks."""
    if not settings.trading_enabled:
        return OrderResult(False, "TRADING_ENABLED is false - order blocked (safe mode)")
    if not MT5_AVAILABLE or not connection.ensure_connected():
        return OrderResult(False, "MT5 not connected")

    ok, why = _validate_trade_permissions()
    if not ok:
        return OrderResult(False, f"permission check failed: {why}")

    ok, why = _validate_market_open(symbol)
    if not ok:
        return OrderResult(False, f"market check failed: {why}")
    ok, why = _validate_spread(symbol)
    if not ok:
        return OrderResult(False, f"spread check failed: {why}")

    if count_open_positions(symbol) >= settings.max_open_positions:
        return OrderResult(False, "max open positions reached")
    if has_open_position(symbol, direction):
        return OrderResult(False, f"duplicate {direction} position exists")

    lot = _normalize_lot(symbol, lot)
    if lot <= 0:
        return OrderResult(False, "invalid lot size after normalisation")

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return OrderResult(False, "no tick price")

    if direction.lower() == "buy":
        order_type = mt5.ORDER_TYPE_BUY
        price = tick.ask
    else:
        order_type = mt5.ORDER_TYPE_SELL
        price = tick.bid

    info = connection.symbol_info(symbol) or {}
    digits = int(info.get("digits", 5) or 5)

    request: dict[str, Any] = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": order_type,
        "price": round(price, digits),
        "sl": round(sl, digits) if sl else 0.0,
        "tp": round(tp, digits) if tp else 0.0,
        "deviation": 20,
        "magic": settings.magic_number,
        "comment": comment[:31],
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": _filling_type(symbol),
    }

    logger.info("Sending {} {} lot={} sl={} tp={}", direction.upper(), symbol, lot, sl, tp)
    result = _send(request)
    if result.ok:
        logger.success("Order placed: {}", result.message)
    else:
        logger.error("Order failed: {}", result.message)
    return result


def open_buy(symbol: str, lot: float, sl: float, tp: float, comment: str = "ai-buy") -> OrderResult:
    """Open a BUY market position."""
    return _open(symbol, "buy", lot, sl, tp, comment)


def open_sell(symbol: str, lot: float, sl: float, tp: float, comment: str = "ai-sell") -> OrderResult:
    """Open a SELL market position."""
    return _open(symbol, "sell", lot, sl, tp, comment)


def _close_position(position: Any) -> OrderResult:
    tick = mt5.symbol_info_tick(position.symbol)
    if tick is None:
        return OrderResult(False, "no tick to close")
    if position.type == mt5.POSITION_TYPE_BUY:
        order_type = mt5.ORDER_TYPE_SELL
        price = tick.bid
    else:
        order_type = mt5.ORDER_TYPE_BUY
        price = tick.ask
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": position.symbol,
        "volume": position.volume,
        "type": order_type,
        "position": position.ticket,
        "price": price,
        "deviation": 20,
        "magic": settings.magic_number,
        "comment": "ai-close",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": _filling_type(position.symbol),
    }
    return _send(request)


def close_all_positions(symbol: str | None = None) -> list[dict[str, Any]]:
    """Close every bot-owned position (optionally for one symbol)."""
    if not MT5_AVAILABLE or not connection.ensure_connected():
        return [{"ok": False, "message": "MT5 not connected"}]
    positions = mt5.positions_get(symbol=symbol or settings.symbol) or []
    results = []
    for p in positions:
        if p.magic != settings.magic_number:
            continue
        res = _close_position(p)
        logger.info("Close ticket {} -> {}", p.ticket, res.message)
        results.append({"ticket": p.ticket, **res.to_dict()})
    return results


def close_profit_positions(symbol: str | None = None) -> list[dict[str, Any]]:
    """Close only positions that are currently in profit."""
    if not MT5_AVAILABLE or not connection.ensure_connected():
        return [{"ok": False, "message": "MT5 not connected"}]
    positions = mt5.positions_get(symbol=symbol or settings.symbol) or []
    results = []
    for p in positions:
        if p.magic != settings.magic_number or p.profit <= 0:
            continue
        res = _close_position(p)
        logger.info("Close profit ticket {} profit={} -> {}", p.ticket, p.profit, res.message)
        results.append({"ticket": p.ticket, "profit": p.profit, **res.to_dict()})
    return results


def update_trailing_stop(symbol: str | None = None) -> list[dict[str, Any]]:
    """Move SL to lock in profit once price has advanced enough.

    Activates once a position is in profit by ``trailing_start_points`` and then
    keeps the SL trailing ``trailing_step_points`` behind price.
    """
    if not settings.trailing_stop:
        return []
    if not MT5_AVAILABLE or not connection.ensure_connected():
        return []

    symbol = symbol or settings.symbol
    info = connection.symbol_info(symbol)
    if info is None:
        return []
    point = info.get("point", 0.0) or 0.0
    if point <= 0:
        return []

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return []

    start_dist = settings.trailing_start_points * point
    step_dist = settings.trailing_step_points * point

    positions = mt5.positions_get(symbol=symbol) or []
    results = []
    for p in positions:
        if p.magic != settings.magic_number:
            continue
        new_sl = None
        if p.type == mt5.POSITION_TYPE_BUY:
            if tick.bid - p.price_open >= start_dist:
                candidate = tick.bid - step_dist
                if p.sl == 0 or candidate > p.sl:
                    new_sl = candidate
        else:  # SELL
            if p.price_open - tick.ask >= start_dist:
                candidate = tick.ask + step_dist
                if p.sl == 0 or candidate < p.sl:
                    new_sl = candidate

        if new_sl is not None:
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": symbol,
                "position": p.ticket,
                "sl": round(new_sl, 5),
                "tp": p.tp,
                "magic": settings.magic_number,
            }
            res = _send(request)
            logger.info("Trailing SL ticket {} -> {} ({})", p.ticket, round(new_sl, 5), res.message)
            results.append({"ticket": p.ticket, "new_sl": round(new_sl, 5), **res.to_dict()})
    return results
