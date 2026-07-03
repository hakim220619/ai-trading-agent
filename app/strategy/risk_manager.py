"""Risk management: position sizing and SL/TP computation."""
from __future__ import annotations

from dataclasses import dataclass

from app.config import settings
from app.mt5.connection import connection
from app.utils.helpers import round_to_step
from app.utils.logger import logger


@dataclass
class TradePlan:
    """A fully-sized trade proposal."""

    direction: str          # "BUY" | "SELL"
    entry: float
    stop_loss: float
    take_profit: float
    lot: float
    risk_money: float
    rr: float

    def to_dict(self) -> dict[str, float | str]:
        return {
            "direction": self.direction,
            "entry": round(self.entry, 5),
            "stop_loss": round(self.stop_loss, 5),
            "take_profit": round(self.take_profit, 5),
            "lot": self.lot,
            "risk_money": round(self.risk_money, 2),
            "rr": self.rr,
        }


def apply_money_limits(
    plan: TradePlan,
    symbol: str,
    stop_loss_money: float = 0.0,
    take_profit_money: float = 0.0,
) -> TradePlan:
    """Convert account-currency SL/TP amounts into broker-aware prices."""
    if stop_loss_money <= 0 and take_profit_money <= 0:
        return plan
    meta = _symbol_meta(symbol)
    value_per_price_unit = plan.lot * meta["tick_value"] / meta["tick_size"]
    if value_per_price_unit <= 0:
        return plan
    min_distance = meta["stops_level"] * meta["point"]
    sl_distance = max(min_distance, stop_loss_money / value_per_price_unit) if stop_loss_money > 0 else abs(plan.entry - plan.stop_loss)
    tp_distance = max(min_distance, take_profit_money / value_per_price_unit) if take_profit_money > 0 else abs(plan.take_profit - plan.entry)
    digits = int(meta["digits"])
    if plan.direction == "BUY":
        plan.stop_loss = round(plan.entry - sl_distance, digits)
        plan.take_profit = round(plan.entry + tp_distance, digits)
    else:
        plan.stop_loss = round(plan.entry + sl_distance, digits)
        plan.take_profit = round(plan.entry - tp_distance, digits)
    plan.risk_money = stop_loss_money if stop_loss_money > 0 else plan.risk_money
    plan.rr = tp_distance / sl_distance if sl_distance > 0 else plan.rr
    return plan


def _symbol_meta(symbol: str) -> dict[str, float]:
    """Return tick value / size / volume constraints with sane fallbacks.

    Falls back to typical XAUUSD-style values when MT5 is unavailable (offline
    backtesting) so sizing math still produces reasonable numbers.
    """
    info = connection.symbol_info(symbol)
    if info:
        return {
            "point": info.get("point", 0.01) or 0.01,
            "tick_value": info.get("trade_tick_value", 1.0) or 1.0,
            "tick_size": info.get("trade_tick_size", 0.01) or 0.01,
            "volume_min": info.get("volume_min", 0.01) or 0.01,
            "volume_max": info.get("volume_max", 100.0) or 100.0,
            "volume_step": info.get("volume_step", 0.01) or 0.01,
            "digits": info.get("digits", 2) or 2,
            "stops_level": info.get("trade_stops_level", 0) or 0,
        }
    # Offline defaults (approx. XAUUSD on a 100oz contract).
    return {
        "point": 0.01,
        "tick_value": 1.0,
        "tick_size": 0.01,
        "volume_min": 0.01,
        "volume_max": 100.0,
        "volume_step": 0.01,
        "digits": 2,
        "stops_level": 0,
    }


def calculate_lot(
    balance: float,
    risk_percent: float,
    entry: float,
    stop_loss: float,
    symbol: str | None = None,
) -> float:
    """Compute lot size so that hitting SL loses ``risk_percent`` of balance.

    lot = risk_money / (sl_distance_in_ticks * tick_value)
    Result is clamped to the broker's volume_min/max and rounded to volume_step.
    """
    symbol = symbol or settings.symbol
    meta = _symbol_meta(symbol)

    risk_money = balance * (risk_percent / 100.0)
    sl_distance = abs(entry - stop_loss)
    if sl_distance <= 0:
        logger.warning("calculate_lot: zero SL distance - using min lot.")
        return meta["volume_min"]

    ticks = sl_distance / meta["tick_size"]
    loss_per_lot = ticks * meta["tick_value"]
    if loss_per_lot <= 0:
        return meta["volume_min"]

    raw_lot = risk_money / loss_per_lot
    lot = round_to_step(raw_lot, meta["volume_step"])
    lot = max(meta["volume_min"], min(meta["volume_max"], lot))
    return round(lot, 2)


def build_trade_plan(
    direction: str,
    entry: float,
    atr_value: float,
    balance: float,
    swing_high: float | None = None,
    swing_low: float | None = None,
    symbol: str | None = None,
    atr_mult: float = 1.5,
    risk_reward: float | None = None,
    fixed_stop_distance: float = 0.0,
    fixed_take_profit_distance: float = 0.0,
) -> TradePlan:
    """Build a full trade plan (SL/TP/lot) for a BUY or SELL.

    SL is placed at the swing low/high if available, otherwise ATR-based.
    TP is derived from the configured risk:reward ratio.
    """
    symbol = symbol or settings.symbol
    direction = direction.upper()
    rr = settings.risk_reward if risk_reward is None else risk_reward
    atr_dist = max(atr_value * atr_mult, entry * 0.0005)  # floor to avoid tiny SL
    stop_dist = fixed_stop_distance if fixed_stop_distance > 0 else atr_dist

    if direction == "BUY":
        sl = entry - stop_dist if fixed_stop_distance > 0 else (swing_low if (swing_low and swing_low < entry) else entry - atr_dist)
        sl = min(sl, entry - atr_dist) if swing_low and fixed_stop_distance <= 0 else sl
        risk = entry - sl
        tp = entry + (fixed_take_profit_distance if fixed_take_profit_distance > 0 else risk * rr)
    else:  # SELL
        sl = entry + stop_dist if fixed_stop_distance > 0 else (swing_high if (swing_high and swing_high > entry) else entry + atr_dist)
        sl = max(sl, entry + atr_dist) if swing_high and fixed_stop_distance <= 0 else sl
        risk = sl - entry
        tp = entry - (fixed_take_profit_distance if fixed_take_profit_distance > 0 else risk * rr)

    # Respect the broker's minimum stop distance and price precision. This
    # prevents otherwise-valid signals from being rejected as "invalid stops".
    meta = _symbol_meta(symbol)
    min_stop_distance = meta["stops_level"] * meta["point"]
    if direction == "BUY":
        if min_stop_distance > 0:
            sl = min(sl, entry - min_stop_distance)
        risk = entry - sl
        tp = entry + (fixed_take_profit_distance if fixed_take_profit_distance > 0 else risk * rr)
    else:
        if min_stop_distance > 0:
            sl = max(sl, entry + min_stop_distance)
        risk = sl - entry
        tp = entry - (fixed_take_profit_distance if fixed_take_profit_distance > 0 else risk * rr)

    digits = int(meta["digits"])
    sl = round(sl, digits)
    tp = round(tp, digits)

    lot = calculate_lot(balance, settings.risk_percent, entry, sl, symbol)
    risk_money = balance * (settings.risk_percent / 100.0)

    plan = TradePlan(
        direction=direction,
        entry=entry,
        stop_loss=sl,
        take_profit=tp,
        lot=lot,
        risk_money=risk_money,
        rr=rr,
    )
    logger.debug("Trade plan: {}", plan.to_dict())
    return plan
