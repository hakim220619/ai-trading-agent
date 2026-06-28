"""Main trading bot + application entry point.

Run the API/dashboard:
    python -m app.main
    # or
    uvicorn app.main:app --host 0.0.0.0 --port 8000

The TradingBot runs its decision loop in a background thread so the FastAPI
dashboard can start/stop it and inspect state live. Default mode is SAFE:
TRADING_ENABLED=false means signals are computed and logged but no live orders
are sent.
"""
from __future__ import annotations

import threading
import time
from typing import Any

import pandas as pd

from app.config import settings
from app.mt5 import order_executor, position_manager
from app.mt5.connection import MT5_AVAILABLE, connection
from app.mt5.market_data import get_candles, get_current_tick
from app.ml.feature_engineering import build_features
from app.strategy.risk_manager import build_trade_plan
from app.strategy.signal_generator import Signal, confirm_multi_timeframe, generate_signal
from app.utils.logger import logger


class TradingBot:
    """Owns the live decision loop and exposes thread-safe controls."""

    # Primary timeframe used for entries; others are for confirmation/context.
    PRIMARY_TF = "M5"

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self.last_signal: Signal | None = None
        self.last_trade_plan: dict[str, object] | None = None
        self.last_order_result: dict[str, object] | None = None
        self.last_run_ts: float = 0.0
        self._last_bar_time: Any = None
        self.confidence_auto: bool = False
        self._last_auto_attempt_ts: float = 0.0
        self.auto_symbols: list[str] = ["BTCUSD", "XAUUSD"]
        self._last_bar_times: dict[str, Any] = {}
        self._last_signals: dict[str, Signal] = {}
        self._last_order_results: dict[str, dict[str, object]] = {}
        self._last_auto_attempts: dict[str, float] = {}

    # --- lifecycle ---------------------------------------------------------
    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, confidence_auto: bool = False) -> bool:
        """Start the background loop. Returns False if already running."""
        with self._lock:
            if self.running:
                if confidence_auto:
                    self.confidence_auto = True
                    logger.info("Confidence auto-trading enabled on running bot.")
                    return True
                logger.warning("Bot already running.")
                return False
            if not MT5_AVAILABLE:
                logger.error("Cannot start trading bot: MetaTrader5 package unavailable.")
                return False
            if not connection.connect():
                logger.error("Cannot start trading bot: MT5 connection/login failed.")
                return False
            self._stop_event.clear()
            self.confidence_auto = confidence_auto
            self.last_order_result = None
            self._last_auto_attempt_ts = 0.0
            self._last_order_results.clear()
            self._last_auto_attempts.clear()
            self._thread = threading.Thread(target=self._run_loop, name="trading-bot", daemon=True)
            self._thread.start()
            logger.success("Trading bot started (trading_enabled={}).", settings.trading_enabled)
            return True

    def stop(self) -> bool:
        """Signal the loop to stop and wait briefly for it to exit."""
        if not self.running:
            return False
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("Trading bot stopped.")
        self.confidence_auto = False
        return True

    def set_symbol(self, symbol: str) -> None:
        """Switch the active market and clear symbol-specific cached state."""
        with self._lock:
            settings.symbol = symbol.upper()
            self._last_bar_time = None
            self.last_signal = None
            self.last_trade_plan = None
            self.last_order_result = None

    def set_auto_symbols(self, symbols: list[str]) -> list[str]:
        """Set the independent markets monitored by Confidence Auto."""
        selected = list(dict.fromkeys(symbol.upper() for symbol in symbols))
        allowed = {"BTCUSD", "XAUUSD"}
        if not selected or any(symbol not in allowed for symbol in selected):
            raise ValueError("pilih minimal satu market: BTCUSD atau XAUUSD")
        with self._lock:
            self.auto_symbols = selected
        logger.info("Confidence auto markets: {}", ", ".join(selected))
        return selected

    # --- main loop ---------------------------------------------------------
    def _run_loop(self) -> None:
        poll = 5  # seconds between checks for a new bar
        while not self._stop_event.is_set():
            try:
                if self.confidence_auto:
                    for symbol in self.auto_symbols:
                        self.tick(symbol)
                else:
                    self.tick(settings.symbol)
            except Exception as exc:  # never let the loop die
                logger.exception("Error in bot loop: {}", exc)
            self._stop_event.wait(poll)

    def _new_bar(self, df: pd.DataFrame, symbol: str) -> bool:
        """Return True only when a new candle has formed since last processed."""
        if df.empty or "time" not in df.columns:
            return True
        latest = df["time"].iloc[-1]
        if latest != self._last_bar_times.get(symbol):
            self._last_bar_time = latest
            self._last_bar_times[symbol] = latest
            return True
        return False

    def tick(self, symbol: str | None = None) -> Signal | None:
        """Run one full decision cycle. Returns the computed Signal."""
        if not MT5_AVAILABLE:
            logger.debug("MT5 unavailable - skipping live tick.")
            return None
        if not connection.ensure_connected():
            logger.warning("MT5 not connected - skipping tick.")
            return None

        symbol = (symbol or settings.symbol).upper()
        df = get_candles(symbol, self.PRIMARY_TF, settings.candles)
        if df.empty:
            return None
        if not self._new_bar(df, symbol):
            # Still manage open positions on every poll (trailing/target).
            position_manager.manage_positions(symbol)
            self._retry_confidence_auto(symbol)
            return self._last_signals.get(symbol)

        # MT5 includes the currently-forming candle as the last row. Decisions
        # must use closed bars so their indicators cannot repaint after entry.
        closed = df.iloc[:-1].copy()
        if closed.empty:
            return None
        frames = self._feature_frames(closed, self.PRIMARY_TF, symbol)
        df = frames[self.PRIMARY_TF]
        sig = confirm_multi_timeframe(
            generate_signal(df),
            frames,
            primary=self.PRIMARY_TF,
        )
        self.last_signal = sig
        self._last_signals[symbol] = sig
        self.last_run_ts = time.time()

        execution_signal = self._confidence_execution_signal(sig, symbol) if self.confidence_auto else sig

        # Manage existing positions, considering the actual execution direction.
        position_manager.manage_positions(symbol, new_signal=execution_signal.action)

        if execution_signal.action in ("BUY", "SELL"):
            self._last_auto_attempt_ts = time.time()
            self._last_auto_attempts[symbol] = self._last_auto_attempt_ts
            self._maybe_enter(execution_signal, df, confidence_sizing=self.confidence_auto, symbol=symbol)
        else:
            logger.info("HOLD - no entry. {}", "; ".join(execution_signal.reasons))

        return sig

    def _confidence_execution_signal(self, sig: Signal, symbol: str) -> Signal:
        """Convert a sufficiently confident ML forecast into an executable signal."""
        confidence = max(float(sig.ml.get("buy", 0.0)), float(sig.ml.get("sell", 0.0)))
        if not sig.ml.get("model") or confidence < 0.65:
            return Signal(price=sig.price, atr=sig.atr, confidence=confidence, ml=sig.ml, levels=sig.levels, reasons=[f"confidence {confidence:.1%} < 65%"])
        tick = get_current_tick(symbol)
        if not tick or time.time() - float(tick.get("time", 0.0)) > 120:
            return Signal(price=sig.price, atr=sig.atr, confidence=confidence, ml=sig.ml, levels=sig.levels, reasons=["auto blocked: market tick inactive / market closed"])
        direction = "BUY" if float(sig.ml.get("buy", 0.0)) >= float(sig.ml.get("sell", 0.0)) else "SELL"
        return Signal(action=direction, price=sig.price, atr=sig.atr, confidence=confidence, ml=sig.ml, levels=sig.levels, reasons=[f"confidence auto {direction} {confidence:.1%}"])

    def _retry_confidence_auto(self, symbol: str) -> None:
        """Retry a rejected confidence order without waiting for another M5 bar."""
        sig = self._last_signals.get(symbol)
        if not self.confidence_auto or sig is None:
            return
        if time.time() - self._last_auto_attempts.get(symbol, 0.0) < 30:
            return
        last_result = self._last_order_results.get(symbol)
        if last_result and bool(last_result.get("ok")):
            return
        execution_signal = self._confidence_execution_signal(sig, symbol)
        if execution_signal.action not in ("BUY", "SELL"):
            return
        if order_executor.count_open_positions(symbol, bot_only=False) >= settings.max_open_positions:
            return
        self._last_auto_attempt_ts = time.time()
        self._last_auto_attempts[symbol] = self._last_auto_attempt_ts
        logger.info("Retrying confidence auto entry {}: {} {:.1%}", symbol, execution_signal.action, execution_signal.confidence)
        self._maybe_enter(execution_signal, pd.DataFrame(), confidence_sizing=True, symbol=symbol)

    def _maybe_enter(self, sig: Signal, df: pd.DataFrame, confidence_sizing: bool = False, symbol: str | None = None) -> None:
        """Size and submit an order for an actionable signal (if enabled)."""
        symbol = (symbol or settings.symbol).upper()
        account = connection.account_info() or {}
        balance = float(account.get("balance", 1000.0))
        tick = get_current_tick(symbol)
        entry = sig.price
        if tick:
            entry = tick["ask"] if sig.action == "BUY" else tick["bid"]

        if confidence_sizing:
            atr_mult = 0.80 if sig.confidence < 0.75 else 1.0 if sig.confidence < 0.85 else 1.20
            risk_reward = 1.0 if sig.confidence < 0.75 else 1.20 if sig.confidence < 0.85 else 1.50
        else:
            atr_mult, risk_reward = 1.5, None
        plan = build_trade_plan(
            direction=sig.action,
            entry=entry,
            atr_value=sig.atr,
            balance=balance,
            swing_high=None if confidence_sizing else sig.levels.get("resistance"),  # type: ignore[arg-type]
            swing_low=None if confidence_sizing else sig.levels.get("support"),       # type: ignore[arg-type]
            atr_mult=atr_mult,
            risk_reward=risk_reward,
            symbol=symbol,
        )
        plan.lot = max(plan.lot, order_executor.configured_minimum_lot(symbol))
        self.last_trade_plan = plan.to_dict()

        if not settings.trading_enabled:
            logger.info(
                "[SAFE MODE] Would {} entry={} lot={} SL={} TP={} but "
                "TRADING_ENABLED=false.",
                sig.action,
                plan.entry,
                plan.lot,
                plan.stop_loss,
                plan.take_profit,
            )
            return

        if order_executor.count_open_positions(symbol, bot_only=False) >= settings.max_open_positions:
            logger.info("Max positions reached for {} - skip entry.", symbol)
            return

        if sig.action == "BUY":
            res = order_executor.open_buy(
                symbol, plan.lot, plan.stop_loss, plan.take_profit,
                enforce_spread=not confidence_sizing,
            )
        else:
            res = order_executor.open_sell(
                symbol, plan.lot, plan.stop_loss, plan.take_profit,
                enforce_spread=not confidence_sizing,
            )
        self.last_order_result = res.to_dict()
        self._last_order_results[symbol] = self.last_order_result
        logger.info("Entry result {}: {} | plan={}", symbol, res.message, plan.to_dict())

    # --- introspection -----------------------------------------------------
    def compute_signal_now(self, timeframe: str | None = None) -> Signal:
        """Compute (without trading) the current signal for inspection."""
        tf = timeframe or self.PRIMARY_TF
        df = get_candles(settings.symbol, tf, settings.candles)
        if len(df) < 2:
            return Signal(reasons=["no data / MT5 unavailable"])
        frames = self._feature_frames(df.iloc[:-1].copy(), tf, settings.symbol)
        return confirm_multi_timeframe(generate_signal(frames[tf]), frames, primary=tf)

    def preview_trade_plan(
        self,
        sig: Signal,
        direction_override: str | None = None,
        atr_mult: float = 1.5,
        risk_reward: float | None = None,
        use_levels: bool = True,
    ) -> dict[str, object] | None:
        """Return the current SL/TP/lot proposal without placing an order."""
        direction = direction_override or sig.action
        if direction not in ("BUY", "SELL"):
            return None
        account = connection.account_info() or {}
        balance = float(account.get("balance", 1000.0))
        tick = get_current_tick(settings.symbol)
        entry = sig.price
        if tick:
            entry = tick["ask"] if direction == "BUY" else tick["bid"]
        plan = build_trade_plan(
            direction=direction,
            entry=entry,
            atr_value=sig.atr,
            balance=balance,
            swing_high=sig.levels.get("resistance") if use_levels else None,  # type: ignore[arg-type]
            swing_low=sig.levels.get("support") if use_levels else None,      # type: ignore[arg-type]
            atr_mult=atr_mult,
            risk_reward=risk_reward,
        )
        plan.lot = max(plan.lot, order_executor.configured_minimum_lot(settings.symbol))
        return plan.to_dict()

    def _feature_frames(
        self,
        primary_df: pd.DataFrame,
        primary_tf: str,
        symbol: str | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Build features for the primary and configured context timeframes."""
        symbol = (symbol or settings.symbol).upper()
        frames = {primary_tf: build_features(primary_df)}
        for timeframe in dict.fromkeys(settings.timeframes):
            if timeframe == primary_tf:
                continue
            context = get_candles(symbol, timeframe, settings.candles)
            if len(context) >= 2:
                frames[timeframe] = build_features(context.iloc[:-1].copy())
        return frames


# Shared singleton used by both the API and CLI entry.
bot = TradingBot()


# --- FastAPI app assembly --------------------------------------------------
def create_app():
    """Build the FastAPI application with routes attached."""
    from fastapi import FastAPI

    from app.api.routes import router

    app = FastAPI(
        title="AI Trading Agent",
        description="Forex/XAUUSD AI trading agent (MT5 + XGBoost). Default mode is SAFE.",
        version="1.0.0",
    )
    app.include_router(router)

    @app.on_event("startup")
    def _auto_start_bot() -> None:
        if settings.auto_start:
            bot.start()

    @app.on_event("shutdown")
    def _stop_bot() -> None:
        if bot.running:
            bot.stop()
        connection.disconnect()

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    logger.info(
        "Starting AI Trading Agent API on {}:{} (trading_enabled={})",
        settings.api_host,
        settings.api_port,
        settings.trading_enabled,
    )
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )
