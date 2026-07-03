"""Alternating M1 recovery scalper driven by floating P/L in account currency."""
from __future__ import annotations

import re
import uuid
import time
from typing import Any

from app.mt5 import order_executor
from app.mt5.position_manager import get_open_positions
from app.runtime_config import get_scalping_setup
from app.utils.logger import logger


_LEGACY_COMMENT = re.compile(r"^m1rec-(\d+)-(buy|sell)$", re.IGNORECASE)
_COMMENT = re.compile(r"^m1r-([a-z0-9]{6})-s(\d+)-([bs])-cf-(\d{5})$", re.IGNORECASE)
_TRUNCATED_COMMENT = re.compile(r"^m1r-([a-z0-9]{6})-s(\d+)-([bs])-?", re.IGNORECASE)
_COMPACT_COMMENT = re.compile(r"^r([a-z0-9]{4})s(\d+)([bs])c(\d{3})$", re.IGNORECASE)
class RecoveryM1Strategy:
    """Keep losing legs open and add alternating counters until basket profit."""

    def __init__(self) -> None:
        self._steps: dict[str, int] = {}
        self._directions: dict[str, str] = {}
        self._cycle_keys: dict[str, str] = {}
        self._confidences: dict[str, float] = {}
        self._pending_open_until: dict[str, float] = {}

    def reset(self, symbol: str) -> None:
        self._steps[symbol] = 1
        self._directions[symbol] = "BUY"
        self._cycle_keys.pop(symbol, None)
        self._confidences.pop(symbol, None)

    def status(self, symbol: str) -> dict[str, Any]:
        step = self._steps.get(symbol, 1)
        setup = get_scalping_setup()
        next_lot = self._lot_for_step(step, setup)
        lots: list[float] = []
        losses: list[float] = []
        for candidate_step in range(1, 21):
            candidate_lot = self._lot_for_step(candidate_step, setup)
            if candidate_lot > setup["max_lot"] + 1e-9:
                break
            lots.append(candidate_lot)
            losses.append(self._loss_for_step(candidate_step, setup))
        trigger_loss_total = sum(losses)
        return {
            "step": step,
            "direction": self._directions.get(symbol, "BUY"),
            "loss_limit_money": self._loss_for_step(step, setup),
            "next_lot": round(next_lot, 2),
            "basket_profit_target": setup["basket_profit_target"],
            "daily_profit_target": setup["daily_profit_target"],
            "confidence_threshold": setup["confidence_threshold"],
            "cycle_key": self._cycle_keys.get(symbol),
            "entry_confidence_pct": round(self._confidences[symbol] * 100, 2) if symbol in self._confidences else None,
            "max_lot": setup["max_lot"],
            "max_lot_step": len(lots),
            "total_lot_at_max": round(sum(lots), 2),
            "trigger_loss_total": round(trigger_loss_total, 2),
            "recommended_safe_capital": round(trigger_loss_total * 3, 2),
        }

    def has_active_positions(self, symbol: str) -> bool:
        return any(self._parse_comment(p.get("comment")) for p in get_open_positions(symbol))

    def tick(self, symbol: str, initial_direction: str | None = None, initial_confidence: float | None = None) -> dict[str, Any]:
        positions = [p for p in get_open_positions(symbol) if self._parse_comment(p.get("comment"))]
        if positions:
            def position_step(item: dict[str, Any]) -> int:
                return self._parse_comment(item.get("comment"))[0]  # type: ignore[index]

            position = max(positions, key=position_step)
            step = position_step(position)
            direction = str(position.get("type_str", "SELL")).upper()
            self._steps[symbol] = step
            self._directions[symbol] = direction
            parsed = self._parse_comment(position.get("comment"))
            if parsed and parsed[2]:
                self._cycle_keys[symbol] = parsed[2]
                self._confidences[symbol] = parsed[3]
            profit = float(position.get("profit", 0.0) or 0.0)
            basket_profit = sum(float(item.get("profit", 0.0) or 0.0) for item in positions)

            setup = get_scalping_setup()
            if basket_profit > setup["basket_profit_target"]:
                return self._close_basket(positions, symbol, basket_profit)

            loss_limit = self._loss_for_step(step, setup)
            if profit <= -loss_limit:
                self._steps[symbol] = step + 1
                self._directions[symbol] = "BUY" if direction == "SELL" else "SELL"
                return self._open(symbol)
            return {
                "action": "HOLD", "profit": profit, "basket_profit": basket_profit,
                "positions": len(positions), "loss_limit_money": loss_limit,
            }

        # No recovery positions means the previous basket is completely gone
        # (including when closed manually from MT5/dashboard). Always discard
        # the old step so the next cycle starts from the configured base lot.
        self.reset(symbol)
        if time.monotonic() < self._pending_open_until.get(symbol, 0.0):
            return {"action": "WAIT_ORDER_SYNC", "ok": True, "message": "menunggu posisi baru tersinkron dari broker"}
        if initial_direction not in {"BUY", "SELL"}:
            return {"action": "WAIT_CONFIDENCE", "ok": True, "message": "confidence BUY/SELL belum lebih dari 50%"}
        self._directions[symbol] = initial_direction
        self._cycle_keys[symbol] = uuid.uuid4().hex[:6]
        self._confidences[symbol] = max(0.0, min(1.0, float(initial_confidence or 0.0)))
        return self._open(symbol)

    def _close_basket(
        self, positions: list[dict[str, Any]], symbol: str, profit: float, action: str = "CLOSE_BASKET_PROFIT",
    ) -> dict[str, Any]:
        results = []
        for position in sorted(positions, key=lambda item: int(item.get("ticket", 0)), reverse=True):
            result = order_executor.close_position_ticket(int(position["ticket"]), symbol)
            results.append({"ticket": int(position["ticket"]), **result.to_dict()})
        ok = bool(results) and all(bool(item["ok"]) for item in results)
        if ok:
            logger.success("Recovery M1 {} basket profit {} closed; cycle reset", symbol, profit)
            self.reset(symbol)
        return {"action": action, "basket_profit": profit, "ok": ok, "results": results}

    def _open(self, symbol: str) -> dict[str, Any]:
        step = self._steps.get(symbol, 1)
        direction = self._directions.get(symbol, "BUY")
        setup = get_scalping_setup()
        lot = self._lot_for_step(step, setup)
        loss_limit = self._loss_for_step(step, setup)
        if lot > setup["max_lot"] + 1e-9:
            return {
                "action": "WAIT_MAX_LOT", "ok": False, "step": step, "requested_lot": lot,
                "max_lot": setup["max_lot"], "message": "lot maksimal scalping tercapai",
            }
        key = (self._cycle_keys.get(symbol) or uuid.uuid4().hex[:4])[:4]
        confidence = self._confidences.get(symbol, 0.0)
        self._cycle_keys[symbol] = key
        confidence_permille = round(confidence * 1_000)
        comment = f"r{key}s{step}{'b' if direction == 'BUY' else 's'}c{confidence_permille:03d}"
        opener = order_executor.open_buy if direction == "BUY" else order_executor.open_sell
        result = opener(
            symbol, lot, 0.0, 0.0, comment=comment, enforce_spread=False,
            allow_duplicate=True, enforce_position_limit=False,
        )
        logger.info("Recovery M1 {} step={} direction={} lot={} limit=${} -> {}", symbol, step, direction, lot, loss_limit, result.message)
        if result.ok:
            self._pending_open_until[symbol] = time.monotonic() + 15.0
        return {
            "action": f"OPEN_{direction}", "step": step, "loss_limit_money": loss_limit,
            "lot": lot, "cycle_key": key.upper(), "confidence_pct": round(confidence * 100, 2), **result.to_dict(),
        }

    @staticmethod
    def _parse_comment(comment: object) -> tuple[int, str, str | None, float] | None:
        text = str(comment or "")
        compact = _COMPACT_COMMENT.match(text)
        if compact:
            return int(compact.group(2)), "BUY" if compact.group(3).lower() == "b" else "SELL", compact.group(1).upper(), int(compact.group(4)) / 1_000
        match = _COMMENT.match(text)
        if match:
            return int(match.group(2)), "BUY" if match.group(3).lower() == "b" else "SELL", match.group(1).upper(), int(match.group(4)) / 10_000
        legacy = _LEGACY_COMMENT.match(text)
        if legacy:
            return int(legacy.group(1)), legacy.group(2).upper(), None, 0.0
        truncated = _TRUNCATED_COMMENT.match(text)
        if truncated:
            return int(truncated.group(2)), "BUY" if truncated.group(3).lower() == "b" else "SELL", truncated.group(1).upper(), 0.0
        return None

    @staticmethod
    def _lot_for_step(step: int, setup: dict[str, float]) -> float:
        if step <= 1:
            lot = setup["base_lot"]
        else:
            lot = setup["second_lot"] * (setup["lot_multiplier"] ** (step - 2))
        return round(lot, 2)

    @staticmethod
    def _loss_for_step(step: int, setup: dict[str, float]) -> float:
        return float(setup["initial_loss_money"] + max(0, step - 1) * setup["loss_increment_money"])
