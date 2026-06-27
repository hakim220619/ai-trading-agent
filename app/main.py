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

    # --- lifecycle ---------------------------------------------------------
    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> bool:
        """Start the background loop. Returns False if already running."""
        with self._lock:
            if self.running:
                logger.warning("Bot already running.")
                return False
            if not MT5_AVAILABLE:
                logger.error("Cannot start trading bot: MetaTrader5 package unavailable.")
                return False
            if not connection.connect():
                logger.error("Cannot start trading bot: MT5 connection/login failed.")
                return False
            self._stop_event.clear()
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
        return True

    # --- main loop ---------------------------------------------------------
    def _run_loop(self) -> None:
        poll = 5  # seconds between checks for a new bar
        while not self._stop_event.is_set():
            try:
                self.tick()
            except Exception as exc:  # never let the loop die
                logger.exception("Error in bot loop: {}", exc)
            self._stop_event.wait(poll)

    def _new_bar(self, df: pd.DataFrame) -> bool:
        """Return True only when a new candle has formed since last processed."""
        if df.empty or "time" not in df.columns:
            return True
        latest = df["time"].iloc[-1]
        if latest != self._last_bar_time:
            self._last_bar_time = latest
            return True
        return False

    def tick(self) -> Signal | None:
        """Run one full decision cycle. Returns the computed Signal."""
        if not MT5_AVAILABLE:
            logger.debug("MT5 unavailable - skipping live tick.")
            return None
        if not connection.ensure_connected():
            logger.warning("MT5 not connected - skipping tick.")
            return None

        df = get_candles(settings.symbol, self.PRIMARY_TF, settings.candles)
        if df.empty:
            return None
        if not self._new_bar(df):
            # Still manage open positions on every poll (trailing/target).
            position_manager.manage_positions(settings.symbol)
            return self.last_signal

        # MT5 includes the currently-forming candle as the last row. Decisions
        # must use closed bars so their indicators cannot repaint after entry.
        closed = df.iloc[:-1].copy()
        if closed.empty:
            return None
        frames = self._feature_frames(closed, self.PRIMARY_TF)
        df = frames[self.PRIMARY_TF]
        sig = confirm_multi_timeframe(
            generate_signal(df),
            frames,
            primary=self.PRIMARY_TF,
        )
        self.last_signal = sig
        self.last_run_ts = time.time()

        # Manage existing positions, considering the fresh signal for reversal.
        position_manager.manage_positions(settings.symbol, new_signal=sig.action)

        if sig.action in ("BUY", "SELL"):
            self._maybe_enter(sig, df)
        else:
            logger.info("HOLD - no entry. {}", "; ".join(sig.reasons))

        return sig

    def _maybe_enter(self, sig: Signal, df: pd.DataFrame) -> None:
        """Size and submit an order for an actionable signal (if enabled)."""
        account = connection.account_info() or {}
        balance = float(account.get("balance", 1000.0))
        tick = get_current_tick(settings.symbol)
        entry = sig.price
        if tick:
            entry = tick["ask"] if sig.action == "BUY" else tick["bid"]

        plan = build_trade_plan(
            direction=sig.action,
            entry=entry,
            atr_value=sig.atr,
            balance=balance,
            swing_high=sig.levels.get("resistance"),  # type: ignore[arg-type]
            swing_low=sig.levels.get("support"),       # type: ignore[arg-type]
        )
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

        if order_executor.count_open_positions(settings.symbol) >= settings.max_open_positions:
            logger.info("Max positions reached - skip entry.")
            return

        if sig.action == "BUY":
            res = order_executor.open_buy(settings.symbol, plan.lot, plan.stop_loss, plan.take_profit)
        else:
            res = order_executor.open_sell(settings.symbol, plan.lot, plan.stop_loss, plan.take_profit)
        self.last_order_result = res.to_dict()
        logger.info("Entry result: {} | plan={}", res.message, plan.to_dict())

    # --- introspection -----------------------------------------------------
    def compute_signal_now(self, timeframe: str | None = None) -> Signal:
        """Compute (without trading) the current signal for inspection."""
        tf = timeframe or self.PRIMARY_TF
        df = get_candles(settings.symbol, tf, settings.candles)
        if len(df) < 2:
            return Signal(reasons=["no data / MT5 unavailable"])
        frames = self._feature_frames(df.iloc[:-1].copy(), tf)
        return confirm_multi_timeframe(generate_signal(frames[tf]), frames, primary=tf)

    def preview_trade_plan(self, sig: Signal) -> dict[str, object] | None:
        """Return the current SL/TP/lot proposal without placing an order."""
        if sig.action not in ("BUY", "SELL"):
            return None
        account = connection.account_info() or {}
        balance = float(account.get("balance", 1000.0))
        tick = get_current_tick(settings.symbol)
        entry = sig.price
        if tick:
            entry = tick["ask"] if sig.action == "BUY" else tick["bid"]
        plan = build_trade_plan(
            direction=sig.action,
            entry=entry,
            atr_value=sig.atr,
            balance=balance,
            swing_high=sig.levels.get("resistance"),  # type: ignore[arg-type]
            swing_low=sig.levels.get("support"),      # type: ignore[arg-type]
        )
        return plan.to_dict()

    def _feature_frames(
        self,
        primary_df: pd.DataFrame,
        primary_tf: str,
    ) -> dict[str, pd.DataFrame]:
        """Build features for the primary and configured context timeframes."""
        frames = {primary_tf: build_features(primary_df)}
        for timeframe in dict.fromkeys(settings.timeframes):
            if timeframe == primary_tf:
                continue
            context = get_candles(settings.symbol, timeframe, settings.candles)
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
