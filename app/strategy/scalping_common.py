"""Shared value objects and market classification for scalping strategies."""
from __future__ import annotations

from dataclasses import dataclass, field

from app.strategy.risk_manager import calculate_lot


@dataclass
class ScalpingSignal:
    action: str = "HOLD"
    price: float = 0.0
    atr: float = 0.0
    reasons: list[str] = field(default_factory=list)


@dataclass
class ScalpingPlan:
    direction: str
    entry: float
    stop_loss: float
    take_profit: float
    lot: float
    risk_money: float
    rr: float


def market_family(symbol: str) -> str:
    """Classify common broker symbol variants without requiring an allow-list."""
    upper = symbol.upper()
    if any(code in upper for code in ("BTC", "ETH", "SOL", "XRP", "DOGE", "LTC")):
        return "crypto"
    if any(code in upper for code in ("XAU", "XAG", "GOLD", "SILVER")):
        return "metals"
    return "standard"


def build_scalping_plan(
    direction: str,
    entry: float,
    atr_value: float,
    balance: float,
    symbol: str,
    risk_percent: float,
    stop_atr: float,
    risk_reward: float,
) -> ScalpingPlan:
    """Create an ATR-normalized plan that scales with any instrument price."""
    direction = direction.upper()
    if direction not in ("BUY", "SELL"):
        raise ValueError("direction must be BUY or SELL")
    distance = max(float(atr_value) * stop_atr, float(entry) * 0.00015)
    stop = entry - distance if direction == "BUY" else entry + distance
    target = entry + distance * risk_reward if direction == "BUY" else entry - distance * risk_reward
    return ScalpingPlan(
        direction=direction,
        entry=entry,
        stop_loss=stop,
        take_profit=target,
        lot=calculate_lot(balance, risk_percent, entry, stop, symbol),
        risk_money=balance * risk_percent / 100.0,
        rr=risk_reward,
    )
