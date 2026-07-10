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
from app.mt5.confidence_metadata import confidence_comment
from app.mt5.connection import MT5_AVAILABLE, connection
from app.mt5.market_data import get_candles, get_current_tick
from app.mt5.daily_limits import daily_summary
from app.ml.feature_engineering import build_features
from app.runtime_config import get_scalping_setup, get_symbol_risk, get_trading_setup
from app.strategy.risk_manager import apply_money_limits, build_trade_plan
from app.strategy.recovery_m1_strategy import RecoveryM1Strategy
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
        self.auto_symbols: list[str] = [settings.symbol]
        self._last_bar_times: dict[str, Any] = {}
        self._last_signals: dict[str, Signal] = {}
        self._last_order_results: dict[str, dict[str, object]] = {}
        self._last_auto_attempts: dict[str, float] = {}
        self.recovery_m1 = RecoveryM1Strategy()

    @property
    def active_strategy(self) -> str:
        return str(get_trading_setup().get("active_strategy", "confidence_m5"))

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
            settings.symbol = symbol.strip()
            self._last_bar_time = None
            self.last_signal = None
            self.last_trade_plan = None
            self.last_order_result = None

    def set_auto_symbols(self, symbols: list[str]) -> list[str]:
        """Set the independent markets monitored by Confidence Auto."""
        selected = list(dict.fromkeys(symbol.strip() for symbol in symbols if symbol.strip()))
        if not selected:
            raise ValueError("pilih minimal satu market")
        with self._lock:
            self.auto_symbols = selected
        logger.info("Confidence auto markets: {}", ", ".join(selected))
        return selected

    # --- main loop ---------------------------------------------------------
    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                if self.confidence_auto:
                    for symbol in self.auto_symbols:
                        self.tick(symbol)
                else:
                    self.tick(settings.symbol)
            except Exception as exc:  # never let the loop die
                logger.exception("Error in bot loop: {}", exc)
            # Recovery counters depend on live floating P/L, so check them every
            # second. Candle-based strategies keep the lighter five-second poll.
            poll = 1 if self.confidence_auto and self.active_strategy == "recovery_m1" else 5
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

    def _check_recovery_daily_caps(
        self,
        symbol: str,
        setup: dict[str, Any],
        has_positions: bool,
    ) -> dict[str, object] | None:
        """Stop Counter Scalping when today's configured P/L cap is reached."""
        daily = daily_summary(bot_only=True)
        daily_profit = float(daily["profit"])
        profit_enabled = bool(setup["daily_profit_target_enabled"])
        profit_target = float(setup["daily_profit_target"])
        loss_enabled = bool(setup.get("daily_loss_limit_enabled", False))
        loss_limit = float(setup.get("daily_loss_limit", 0.0))

        action = ""
        message = ""
        if profit_enabled and profit_target > 0 and daily_profit >= profit_target:
            action = "WAIT_DAILY_PROFIT_TARGET"
            message = f"target profit harian tercapai ({daily['profit']}/{profit_target}); auto trade dihentikan"
        elif loss_enabled and loss_limit > 0 and daily_profit <= -loss_limit:
            action = "WAIT_DAILY_LOSS_LIMIT"
            message = f"batas loss harian tercapai ({daily['profit']}/-{loss_limit}); auto trade dihentikan"
        else:
            return None

        close_results = order_executor.close_all_positions(symbol) if has_positions else []
        self.confidence_auto = False
        self._stop_event.set()
        return {
            "action": action, "ok": True, "message": message,
            "daily_profit": daily_profit, "daily_profit_target": profit_target,
            "daily_loss_limit": loss_limit, "closed_positions": close_results,
            "auto_trade_stopped": True,
        }

    def _trading_hour_allowed(self) -> tuple[bool, int, dict[str, Any]]:
        setup = get_trading_setup()
        current_hour = time.localtime().tm_hour
        hours = setup.get("trading_hours", [True] * 24)
        allowed = True
        if bool(setup.get("trading_hours_enabled", False)):
            allowed = bool(isinstance(hours, list) and len(hours) > current_hour and hours[current_hour])
        return allowed, current_hour, setup

    def tick(self, symbol: str | None = None) -> Signal | None:
        """Run one full decision cycle. Returns the computed Signal."""
        if not MT5_AVAILABLE:
            logger.debug("MT5 unavailable - skipping live tick.")
            return None
        if not connection.ensure_connected():
            logger.warning("MT5 not connected - skipping tick.")
            return None

        symbol = (symbol or settings.symbol).strip()
        if self.confidence_auto and self.active_strategy == "recovery_m1":
            initial_direction = None
            has_recovery_positions = self.recovery_m1.has_active_positions(symbol)
            scalping_setup = get_scalping_setup(symbol)
            hour_allowed, current_hour, _setup = self._trading_hour_allowed()
            daily_cap_result = self._check_recovery_daily_caps(symbol, scalping_setup, has_recovery_positions)
            if daily_cap_result:
                self.last_order_result = daily_cap_result
                self._last_order_results[symbol] = daily_cap_result
                self.last_run_ts = time.time()
                return None
            if not has_recovery_positions:
                if not hour_allowed:
                    result = {
                        "action": "WAIT_TRADING_HOUR", "ok": True,
                        "message": f"jam trading {current_hour:02d}:00-{(current_hour + 1) % 24:02d}:00 non aktif",
                        "hour": current_hour,
                    }
                    self.last_order_result = result
                    self._last_order_results[symbol] = result
                    self.last_run_ts = time.time()
                    return None
                sig = self.compute_signal_now("M1", symbol)
                self.last_signal = sig
                self._last_signals[symbol] = sig
                buy_confidence = float(sig.ml.get("buy", 0.0) or 0.0)
                sell_confidence = float(sig.ml.get("sell", 0.0) or 0.0)
                threshold = float(scalping_setup["confidence_threshold"])
                if buy_confidence > threshold and buy_confidence > sell_confidence:
                    initial_direction = "BUY"
                elif sell_confidence > threshold and sell_confidence > buy_confidence:
                    initial_direction = "SELL"
            initial_confidence = None
            if initial_direction == "BUY":
                initial_confidence = buy_confidence
            elif initial_direction == "SELL":
                initial_confidence = sell_confidence
            result = self.recovery_m1.tick(
                symbol,
                initial_direction=initial_direction,
                initial_confidence=initial_confidence,
                allow_new_entries=hour_allowed,
            )
            self.last_order_result = result
            self._last_order_results[symbol] = result
            self.last_run_ts = time.time()
            return None

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
            hour_allowed, current_hour, _setup = self._trading_hour_allowed()
            if not hour_allowed:
                result = {
                    "action": "WAIT_TRADING_HOUR", "ok": True,
                    "message": f"jam trading {current_hour:02d}:00-{(current_hour + 1) % 24:02d}:00 non aktif",
                    "hour": current_hour,
                }
                self.last_order_result = result
                self._last_order_results[symbol] = result
                logger.info("Auto entry {} blocked by trading hour {}", symbol, current_hour)
                return sig
            self._last_auto_attempt_ts = time.time()
            self._last_auto_attempts[symbol] = self._last_auto_attempt_ts
            self._maybe_enter(execution_signal, df, confidence_sizing=self.confidence_auto, symbol=symbol)
        else:
            logger.info("HOLD - no entry. {}", "; ".join(execution_signal.reasons))

        return sig

    def _confidence_execution_signal(self, sig: Signal, symbol: str) -> Signal:
        """Convert a sufficiently confident ML forecast into an executable signal."""
        direction, confidence = self.confidence_recommendation(sig)
        threshold = float(get_trading_setup()["confidence_threshold"])
        if not sig.ml.get("model") or confidence < threshold:
            return Signal(price=sig.price, atr=sig.atr, confidence=confidence, ml=sig.ml, levels=sig.levels, reasons=[f"confidence {confidence:.1%} < {threshold:.1%}"])
        tick = get_current_tick(symbol)
        if not tick or time.time() - float(tick.get("time", 0.0)) > 120:
            return Signal(price=sig.price, atr=sig.atr, confidence=confidence, ml=sig.ml, levels=sig.levels, reasons=["auto blocked: market tick inactive / market closed"])
        entry_price = float(tick["ask"] if direction == "BUY" else tick["bid"])
        sr_ok, sr_reason = self.confidence_support_resistance_check(sig, direction, entry_price)
        if not sr_ok:
            return Signal(price=sig.price, atr=sig.atr, confidence=confidence, ml=sig.ml, levels=sig.levels, reasons=[sr_reason])
        return Signal(action=direction, price=sig.price, atr=sig.atr, confidence=confidence, ml=sig.ml, levels=sig.levels, reasons=[f"confidence auto {direction} {confidence:.1%}", sr_reason])

    def confidence_support_resistance_check(
        self,
        sig: Signal,
        direction: str,
        entry_price: float | None = None,
    ) -> tuple[bool, str]:
        """Gate Confidence Auto entries to BUY support / SELL resistance areas."""
        price = float(entry_price or sig.price or 0.0)
        if price <= 0:
            return False, "auto blocked: harga entry tidak valid untuk cek support/resistance"
        levels = sig.levels or {}
        tolerance = 0.003
        if direction == "BUY":
            support = levels.get("support")
            if support is None:
                return False, "auto blocked: BUY butuh area support terdeteksi"
            distance = abs(price - float(support)) / price
            if distance > tolerance:
                return False, f"auto blocked: BUY belum dekat support ({distance:.2%} > {tolerance:.2%})"
            return True, f"BUY dekat support {float(support):.5f} ({distance:.2%})"
        resistance = levels.get("resistance")
        if resistance is None:
            return False, "auto blocked: SELL butuh area resistance terdeteksi"
        distance = abs(float(resistance) - price) / price
        if distance > tolerance:
            return False, f"auto blocked: SELL belum dekat resistance ({distance:.2%} > {tolerance:.2%})"
        return True, f"SELL dekat resistance {float(resistance):.5f} ({distance:.2%})"

    def confidence_recommendation(self, sig: Signal) -> tuple[str, float]:
        """Return the ML side and probability used by Confidence Auto."""
        buy = float(sig.ml.get("buy", 0.0) or 0.0)
        sell = float(sig.ml.get("sell", 0.0) or 0.0)
        return ("BUY", buy) if buy >= sell else ("SELL", sell)

    def directional_confidence(self, sig: Signal, direction: str) -> float:
        """Return confidence for the selected direction, not the opposite side."""
        key = "buy" if direction.upper() == "BUY" else "sell"
        value = sig.ml.get(key)
        if value is None:
            return float(sig.confidence or 0.0)
        return max(0.0, min(1.0, float(value or 0.0)))

    def confidence_risk_params(self, confidence: float) -> tuple[float, float]:
        """Risk profile used by both confidence preview and live entry."""
        confidence = max(0.0, min(1.0, float(confidence or 0.0)))
        atr_mult = 0.80 if confidence < 0.75 else 1.0 if confidence < 0.85 else 1.20
        risk_reward = 1.0 if confidence < 0.75 else 1.20 if confidence < 0.85 else 1.50
        return atr_mult, risk_reward

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
        hour_allowed, current_hour, _setup = self._trading_hour_allowed()
        if not hour_allowed:
            logger.info("Retry confidence auto {} blocked by trading hour {}", symbol, current_hour)
            return
        execution_signal = self._confidence_execution_signal(sig, symbol)
        if execution_signal.action not in ("BUY", "SELL"):
            return
        if order_executor.count_open_positions(symbol, bot_only=False) >= int(get_trading_setup()["max_open_positions"]):
            return
        self._last_auto_attempt_ts = time.time()
        self._last_auto_attempts[symbol] = self._last_auto_attempt_ts
        logger.info("Retrying confidence auto entry {}: {} {:.1%}", symbol, execution_signal.action, execution_signal.confidence)
        self._maybe_enter(execution_signal, pd.DataFrame(), confidence_sizing=True, symbol=symbol)

    def _maybe_enter(self, sig: Signal, df: pd.DataFrame, confidence_sizing: bool = False, symbol: str | None = None) -> None:
        """Size and submit an order for an actionable signal (if enabled)."""
        symbol = (symbol or settings.symbol).strip()
        account = connection.account_info() or {}
        balance = float(account.get("balance", 1000.0))
        tick = get_current_tick(symbol)
        entry = sig.price
        if tick:
            entry = tick["ask"] if sig.action == "BUY" else tick["bid"]

        atr_mult, risk_reward = self.confidence_risk_params(sig.confidence) if confidence_sizing else (1.5, None)
        risk_config = get_symbol_risk(symbol)
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
        plan = apply_money_limits(plan, symbol, risk_config["stop_loss_money"], risk_config["take_profit_money"])
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

        if order_executor.count_open_positions(symbol, bot_only=False) >= int(get_trading_setup()["max_open_positions"]):
            logger.info("Max positions reached for {} - skip entry.", symbol)
            return

        if sig.action == "BUY":
            res = order_executor.open_buy(
                symbol, plan.lot, plan.stop_loss, plan.take_profit,
                comment=confidence_comment(sig.confidence, sig.action) if confidence_sizing else "ai-buy",
                enforce_spread=not confidence_sizing,
            )
        else:
            res = order_executor.open_sell(
                symbol, plan.lot, plan.stop_loss, plan.take_profit,
                comment=confidence_comment(sig.confidence, sig.action) if confidence_sizing else "ai-sell",
                enforce_spread=not confidence_sizing,
            )
        self.last_order_result = res.to_dict()
        self._last_order_results[symbol] = self.last_order_result
        logger.info("Entry result {}: {} | plan={}", symbol, res.message, plan.to_dict())

    # --- introspection -----------------------------------------------------
    def compute_signal_now(self, timeframe: str | None = None, symbol: str | None = None) -> Signal:
        """Compute (without trading) the current signal for inspection."""
        tf = timeframe or self.PRIMARY_TF
        symbol = (symbol or settings.symbol).strip()
        df = get_candles(symbol, tf, settings.candles)
        if len(df) < 2:
            return Signal(reasons=["no data / MT5 unavailable"])
        frames = self._feature_frames(df.iloc[:-1].copy(), tf, symbol)
        return confirm_multi_timeframe(generate_signal(frames[tf]), frames, primary=tf)

    def preview_trade_plan(
        self,
        sig: Signal,
        direction_override: str | None = None,
        atr_mult: float = 1.5,
        risk_reward: float | None = None,
        use_levels: bool = True,
        symbol: str | None = None,
    ) -> dict[str, object] | None:
        """Return the current SL/TP/lot proposal without placing an order."""
        direction = direction_override or sig.action
        if direction not in ("BUY", "SELL"):
            return None
        symbol = (symbol or settings.symbol).strip()
        account = connection.account_info() or {}
        balance = float(account.get("balance", 1000.0))
        tick = get_current_tick(symbol)
        entry = sig.price
        if tick:
            entry = tick["ask"] if direction == "BUY" else tick["bid"]
        risk_config = get_symbol_risk(symbol)
        plan = build_trade_plan(
            direction=direction,
            entry=entry,
            atr_value=sig.atr,
            balance=balance,
            swing_high=sig.levels.get("resistance") if use_levels else None,  # type: ignore[arg-type]
            swing_low=sig.levels.get("support") if use_levels else None,      # type: ignore[arg-type]
            atr_mult=atr_mult,
            risk_reward=risk_reward,
            symbol=symbol,
        )
        plan.lot = max(plan.lot, order_executor.configured_minimum_lot(symbol))
        plan = apply_money_limits(plan, symbol, risk_config["stop_loss_money"], risk_config["take_profit_money"])
        return plan.to_dict()

    def _feature_frames(
        self,
        primary_df: pd.DataFrame,
        primary_tf: str,
        symbol: str | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Build features for the primary and configured context timeframes."""
        symbol = (symbol or settings.symbol).strip()
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
    import asyncio
    from urllib.parse import urlencode
    from fastapi import FastAPI, Request
    from fastapi.responses import Response

    from app.api.routes import router
    from app.account_manager import account_manager, is_account_worker

    app = FastAPI(
        title="AI Trading Agent",
        description="Forex/XAUUSD AI trading agent (MT5 + XGBoost/LightGBM ensemble). Default mode is SAFE.",
        version="1.0.0",
    )
    app.include_router(router)

    if not is_account_worker():
        @app.middleware("http")
        async def route_account_worker(request: Request, call_next):
            account_id = request.query_params.get("account_id", "default")
            if account_id == "default" or request.url.path in {"/", "/accounts", "/account/login"}:
                return await call_next(request)
            worker = account_manager.get(account_id)
            if worker is None:
                return Response('{"detail":"akun tidak aktif"}', status_code=404, media_type="application/json")
            query = urlencode([(key, value) for key, value in request.query_params.multi_items() if key != "account_id"])
            path = request.url.path + (f"?{query}" if query else "")
            body = await request.body()
            headers = {"Content-Type": request.headers.get("content-type", "application/json"), "Accept": "application/json"}
            try:
                status, content, content_type = await asyncio.to_thread(
                    account_manager.request, account_id, request.method, path, body or None, headers
                )
            except Exception as exc:
                return Response(f'{{"detail":"worker akun gagal: {str(exc)}"}}', status_code=502, media_type="application/json")
            return Response(content, status_code=status, media_type=content_type.split(";", 1)[0])

    @app.on_event("startup")
    def _auto_start_bot() -> None:
        if not is_account_worker():
            account_manager.restore()
        if settings.auto_start:
            bot.start()

    @app.on_event("shutdown")
    def _stop_bot() -> None:
        if bot.running:
            bot.stop()
        connection.disconnect()
        if not is_account_worker():
            account_manager.shutdown()

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
