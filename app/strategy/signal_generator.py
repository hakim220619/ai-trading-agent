"""Rule-based + ML signal generation."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from app.config import settings
from app.ml.predict import predict_signal
from app.strategy import support_resistance as sr
from app.strategy.indicators import latest_indicator_snapshot
from app.utils.logger import logger


@dataclass
class Signal:
    """Trading decision with full reasoning trail."""

    action: str = "HOLD"             # BUY | SELL | HOLD
    confidence: float = 0.0          # ML probability backing the action
    price: float = 0.0
    atr: float = 0.0
    reasons: list[str] = field(default_factory=list)
    levels: dict[str, object] = field(default_factory=dict)
    ml: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "action": self.action,
            "confidence": round(self.confidence, 4),
            "price": self.price,
            "atr": round(self.atr, 5),
            "reasons": self.reasons,
            "levels": self.levels,
            "ml": self.ml,
        }


def _spread_ok() -> tuple[bool, float | None]:
    """Check the live spread against the configured maximum.

    When MT5 is unavailable (offline backtest), spread is treated as OK.
    """
    from app.mt5.connection import MT5_AVAILABLE, connection

    if not MT5_AVAILABLE:
        return True, None
    spread = connection.get_spread_points()
    if spread is None:
        return True, None
    return spread <= settings.max_spread_points, spread


def generate_signal(df: pd.DataFrame, features: dict[str, float] | None = None) -> Signal:
    """Combine indicator rules, support/resistance and ML to decide BUY/SELL/HOLD.

    ``df`` must already contain indicator columns (see indicators.add_indicators).
    """
    sig = Signal()
    if df is None or df.empty:
        sig.reasons.append("no data")
        return sig

    snap = latest_indicator_snapshot(df)
    if not snap:
        sig.reasons.append("indicators not ready")
        return sig

    price = snap["close"]
    sig.price = price
    sig.atr = snap.get("atr", 0.0)

    levels = sr.detect_levels(df)
    sig.levels = levels.to_dict()

    # ML probabilities.
    feats = features if features is not None else snap
    ml = predict_signal(feats)
    sig.ml = ml
    prob_buy = ml.get("buy", 0.5)
    prob_sell = ml.get("sell", 0.5)

    spread_ok, spread_val = _spread_ok()
    if not spread_ok:
        sig.reasons.append(f"spread too high ({spread_val})")
        return sig  # HOLD - never trade on bad spread

    ema20 = snap.get("ema20", price)
    ema50 = snap.get("ema50", price)
    ema200 = snap.get("ema200", price)
    rsi_v = snap.get("rsi", 50.0)

    threshold = settings.ml_prob_threshold

    # --- BUY rules ---
    buy_checks = {
        "ema20>ema50": ema20 > ema50,
        "price>ema200": price > ema200,
        "rsi 50-70": 50 <= rsi_v <= 70,
        "near support": sr.is_near_support(df, levels),
        f"ml_buy>{threshold}": prob_buy >= threshold,
    }
    # --- SELL rules ---
    sell_checks = {
        "ema20<ema50": ema20 < ema50,
        "price<ema200": price < ema200,
        "rsi 30-50": 30 <= rsi_v <= 50,
        "near resistance": sr.is_near_resistance(df, levels),
        f"ml_sell>{threshold}": prob_sell >= threshold,
    }

    buy_passed = [k for k, v in buy_checks.items() if v]
    sell_passed = [k for k, v in sell_checks.items() if v]

    if all(buy_checks.values()):
        sig.action = "BUY"
        sig.confidence = prob_buy
        sig.reasons = [f"BUY ok: {', '.join(buy_passed)}", "spread ok"]
    elif all(sell_checks.values()):
        sig.action = "SELL"
        sig.confidence = prob_sell
        sig.reasons = [f"SELL ok: {', '.join(sell_passed)}", "spread ok"]
    else:
        sig.action = "HOLD"
        sig.confidence = max(prob_buy, prob_sell)
        sig.reasons = [
            f"BUY {len(buy_passed)}/5 [{', '.join(buy_passed) or '-'}]",
            f"SELL {len(sell_passed)}/5 [{', '.join(sell_passed) or '-'}]",
        ]

    logger.info("Signal: {} conf={:.2f} | {}", sig.action, sig.confidence, "; ".join(sig.reasons))
    return sig
