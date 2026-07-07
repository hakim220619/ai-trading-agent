"""FastAPI routes for monitoring and control."""
from __future__ import annotations

import os
import re
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

from app.api.schemas import (
    AccountResponse,
    ActionResponse,
    AutoMarketsRequest,
    BacktestRequest,
    BulkLevelRequest,
    ExportRequest,
    ManualTradeRequest,
    MT5LoginRequest,
    PositionsResponse,
    ScalpingSetupRequest,
    SignalResponse,
    SymbolRequest,
    SymbolRiskConfigRequest,
    StatusResponse,
    TrainRequest,
    TradeHistoryResponse,
    TradingSetupRequest,
)
from app.config import settings
from app.mt5 import order_executor, position_manager
from app.mt5.connection import MT5_AVAILABLE, connection
from app.mt5.daily_limits import daily_summary
from app.mt5.confidence_metadata import confidence_comment
from app.mt5.market_data import get_candles
from app.ml.predict import get_model
from app.runtime_config import get_all_scalping_setups, get_all_symbol_risk, get_scalping_setup, get_trading_setup, save_scalping_setup, save_symbol_risk, save_trading_setup
from app.strategy import support_resistance as sr
from app.utils.logger import logger
from app.account_manager import account_manager, is_account_worker

router = APIRouter()


def _get_bot():
    """Lazy import to avoid circular import with app.main."""
    from app.main import bot

    return bot


@router.get("/", response_class=HTMLResponse, tags=["dashboard"])
def dashboard() -> str:
    """Responsive Bootstrap monitoring dashboard."""
    return Path(__file__).with_name("dashboard.html").read_text(encoding="utf-8")


@router.get("/strategies", tags=["monitor"])
def strategies() -> dict:
    """Describe what is active live versus available only for backtesting."""
    setup = get_trading_setup()
    return {
        "active_symbol": settings.symbol,
        "auto_symbols": _get_bot().auto_symbols,
        "active_strategy": setup["active_strategy"],
        "active_timeframe": "M1" if setup["active_strategy"] == "recovery_m1" else "M5",
        "confidence_auto": _get_bot().confidence_auto,
        "choices": [
            {"id": "confidence_m5", "name": "Confidence M5", "detail": "Ensemble XGBoost + LightGBM dengan technical/MTF dan pengaturan risiko yang ada."},
            {"id": "recovery_m1", "name": "Counter Basket Scalping M1", "detail": "Entry awal mengikuti confidence M1; lot 0.01 dengan batas loss $3, lalu counter 0.03 dan lot berikutnya ×2: 0.06, 0.12, 0.24, ..."},
        ],
        "live": ([
            {"name": "Counter Basket Scalping M1", "detail": "Entry awal BUY/SELL hanya jika confidence M1 salah satu sisi > 50%. Lot awal 0.01 (loss $3), counter 0.03, kemudian lot dikali 2: 0.06, 0.12, 0.24, ..."},
            {"name": "Risk & Execution", "detail": "Lot memakai minimum lot market; tanpa SL/TP harga, tetap tunduk pada batas trading harian."},
        ] if setup["active_strategy"] == "recovery_m1" else [
            {"name": "Technical Trend", "detail": "EMA20/EMA50/EMA200 + RSI"},
            {"name": "Support & Resistance", "detail": "Swing, breakout, retest, BOS, dan CHoCH"},
            {"name": "Ensemble ML Confidence", "detail": f"Probabilitas gabungan XGBoost + LightGBM; BUY/SELL jika confidence >= {float(setup['confidence_threshold']):.0%}."},
            {"name": "Multi-Timeframe", "detail": "Konfirmasi konteks M15 dan H1"},
            {"name": "Risk & Execution", "detail": "ATR SL/TP adaptif, maksimal 3 posisi, trailing SL mengunci profit setiap kenaikan $1"},
        ]),
        "backtest_only": [
            {"name": "Combined Scalping M1", "module": "scalping_m1_strategy"},
            {"name": "MA Crossover M1", "module": "scalping_ma_m1_strategy"},
            {"name": "Supply & Demand M1", "module": "scalping_snd_m1_strategy"},
            {"name": "RSI Reversal", "module": "rsi_strategy"},
            {"name": "Combined Scalping M5", "module": "scalping_m5_strategy"},
            {"name": "Universal Day Trade", "module": "day_trade_strategy"},
        ],
    }


@router.get("/status", response_model=StatusResponse, tags=["monitor"])
def status() -> StatusResponse:
    bot = _get_bot()
    setup = get_trading_setup()
    connected = MT5_AVAILABLE and connection.ensure_connected()
    account_info = connection.account_info() if connected else None
    positions = position_manager.get_open_positions(bot_only=False) if connected else []
    total_profit = round(sum(float(p.get("profit", 0.0)) for p in positions), 2)
    daily = daily_summary(bot_only=True) if connected else {"profit": 0.0, "lot": 0.0}
    trade_status = connection.trading_status() if connected else {
        "terminal_trade_allowed": False,
        "account_trade_allowed": False,
        "trade_api_disabled": True,
    }
    health = connection.connection_health(settings.symbol) if connected else {}
    return StatusResponse(
        running=bot.running,
        trading_enabled=settings.trading_enabled,
        mt5_connected=connected,
        broker_connected=bool(health.get("broker_connected", False)),
        connection_stable=bool(health.get("stable", False)),
        ping_ms=health.get("ping_ms"),
        ping_jitter_ms=health.get("jitter_ms"),
        tick_age_seconds=health.get("tick_age_seconds"),
        symbol=settings.symbol,
        timeframes=settings.timeframes,
        open_positions=len(positions),
        model_loaded=get_model() is not None,
        account_balance=(round(float(account_info["balance"]), 2) if account_info else None),
        account_equity=(round(float(account_info["equity"]), 2) if account_info else None),
        account_currency=(str(account_info["currency"]) if account_info else None),
        total_profit=total_profit,
        confidence_auto=bot.confidence_auto,
        active_strategy=bot.active_strategy,
        strategy_status=(bot.recovery_m1.status(settings.symbol) if bot.active_strategy == "recovery_m1" else {}),
        auto_symbols=bot.auto_symbols,
        max_open_positions=int(setup["max_open_positions"]),
        confidence_threshold=float(setup["confidence_threshold"]),
        trailing_stop=bool(setup["trailing_stop"]),
        trailing_profit_step_money=float(setup["trailing_profit_step_money"]),
        daily_profit_limit_enabled=bool(setup["daily_profit_limit_enabled"]),
        daily_profit_limit_money=float(setup["daily_profit_limit_money"]),
        daily_lot_limit_enabled=bool(setup["daily_lot_limit_enabled"]),
        daily_lot_limit=float(setup["daily_lot_limit"]),
        daily_profit_today=float(daily["profit"]),
        daily_lot_today=float(daily["lot"]),
        **trade_status,
    )


@router.post("/symbol", response_model=ActionResponse, tags=["control"])
def select_symbol(req: SymbolRequest) -> ActionResponse:
    bot = _get_bot()
    symbol = req.symbol.strip()
    if not connection.symbol_info(symbol):
        return ActionResponse(ok=False, message=f"symbol {symbol} tidak tersedia di MT5")
    bot.set_symbol(symbol)
    return ActionResponse(
        ok=True,
        message=f"market aktif diubah ke {symbol}",
        detail={"symbol": symbol, "bot_running": bot.running},
    )


@router.get("/risk-config", tags=["monitor"])
def risk_config() -> dict:
    return {
        "symbols": get_all_symbol_risk(),
        "trading_setup": get_trading_setup(),
        "daily_summary": daily_summary(bot_only=True),
        "unit": "account_currency",
        "zero_means": "automatic_atr",
    }


@router.get("/markets", tags=["monitor"])
def markets(only_open: bool = True, search: str | None = None, limit: int = 500, max_tick_age_minutes: int = 180) -> dict:
    """List broker markets from MT5 with a best-effort open status."""
    limit = max(1, min(limit, 2000))
    max_tick_age_minutes = max(1, min(max_tick_age_minutes, 1440))
    rows = connection.list_markets(
        only_open=only_open,
        search=search,
        limit=limit,
        max_tick_age_minutes=max_tick_age_minutes,
    )
    return {
        "connected": bool(connection.connected),
        "only_open": only_open,
        "search": search,
        "count": len(rows),
        "markets": rows,
    }


@router.get("/candles", tags=["monitor"])
def candles(symbol: str | None = None, timeframe: str = "M5", count: int = 120) -> dict:
    selected_symbol = symbol.strip() if symbol else settings.symbol
    timeframe = timeframe.upper()
    count = max(30, min(count, 500))
    df = get_candles(selected_symbol, timeframe, count)
    if df.empty:
        return {"symbol": selected_symbol, "timeframe": timeframe, "candles": [], "levels": {}}
    levels = sr.detect_levels(df).to_dict()
    close = df["close"].astype(float)
    delta = close.diff()
    average_gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    average_loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    relative_strength = average_gain / average_loss.replace(0, float("nan"))
    rsi_values = 100 - (100 / (1 + relative_strength))
    rsi_values = rsi_values.mask((average_loss == 0) & (average_gain > 0), 100.0).fillna(50.0)
    rows = []
    chart_df = df.copy()
    chart_df["rsi"] = rsi_values
    for row in chart_df.tail(count).to_dict("records"):
        timestamp = row.get("time")
        if hasattr(timestamp, "timestamp"):
            timestamp = int(timestamp.timestamp())
        rows.append({
            "time": timestamp,
            "open": float(row.get("open", 0.0) or 0.0),
            "high": float(row.get("high", 0.0) or 0.0),
            "low": float(row.get("low", 0.0) or 0.0),
            "close": float(row.get("close", 0.0) or 0.0),
            "rsi": float(row.get("rsi", 50.0) or 50.0),
        })
    return {
        "symbol": selected_symbol,
        "timeframe": timeframe,
        "candles": rows,
        "levels": levels,
        "strategies": _chart_strategy_signals(df, selected_symbol, timeframe),
    }


def _chart_strategy_signals(df, symbol: str, timeframe: str) -> list[dict[str, object]]:
    """Evaluate every standalone strategy for the chart's latest candle."""
    from app.strategy.day_trade_strategy import generate_day_trade_signal, get_day_trade_preset, prepare_day_trade_features
    from app.strategy.rsi_strategy import generate_rsi_signal, prepare_rsi_features
    from app.strategy.scalping_m1_strategy import generate_m1_signal, prepare_m1_features
    from app.strategy.scalping_m5_strategy import generate_m5_signal, prepare_m5_features
    from app.strategy.scalping_ma_m1_strategy import generate_ma_m1_signal, prepare_ma_m1_features
    from app.strategy.scalping_snd_m1_strategy import generate_snd_m1_signal, prepare_snd_m1_features

    spread = float(df.iloc[-1].get("spread", 0.0) or 0.0)
    maximum = max(float(settings.max_spread_points), spread)
    definitions = [
        ("Combined M1", "M1", lambda: generate_m1_signal(prepare_m1_features(df).iloc[-1], symbol, spread, maximum)),
        ("EMA Crossover M1", "M1", lambda: generate_ma_m1_signal(prepare_ma_m1_features(df).iloc[-1], symbol, spread, maximum)),
        ("Supply/Demand M1", "M1", lambda: generate_snd_m1_signal(prepare_snd_m1_features(df).iloc[-1], symbol, spread, maximum)),
        ("RSI Reversal", timeframe, lambda: generate_rsi_signal(prepare_rsi_features(df).iloc[-1], symbol, spread, maximum)),
        ("Combined M5", "M5", lambda: generate_m5_signal(prepare_m5_features(df, symbol).iloc[-1], symbol, spread, maximum)),
        ("Universal Day Trade", timeframe, lambda: generate_day_trade_signal(
            prepare_day_trade_features(df, get_day_trade_preset(symbol).breakout_period).iloc[-1],
            spread,
            maximum,
            get_day_trade_preset(symbol),
        )),
    ]
    results: list[dict[str, object]] = []
    for name, intended_timeframe, evaluate in definitions:
        try:
            signal = evaluate()
            results.append({
                "name": name,
                "action": signal.action,
                "reason": signal.reasons[0] if signal.reasons else "tidak ada alasan",
                "timeframe": intended_timeframe,
                "timeframe_match": intended_timeframe == timeframe,
            })
        except Exception as exc:
            logger.warning("Chart strategy {} failed: {}", name, exc)
            results.append({"name": name, "action": "WAIT", "reason": "indikator belum siap", "timeframe": intended_timeframe, "timeframe_match": False})
    return results


@router.post("/risk-config", response_model=ActionResponse, tags=["control"])
def update_risk_config(req: SymbolRiskConfigRequest) -> ActionResponse:
    values = save_symbol_risk(req.symbol, req.stop_loss_money, req.take_profit_money)
    return ActionResponse(
        ok=True,
        message=f"konfigurasi SL/TP {req.symbol} berhasil disimpan",
        detail={"symbol": req.symbol, **values},
    )


@router.post("/trading-setup", response_model=ActionResponse, tags=["control"])
def update_trading_setup(req: TradingSetupRequest) -> ActionResponse:
    payload = req.model_dump()
    if payload.get("active_strategy") is None:
        payload.pop("active_strategy", None)
    values = save_trading_setup(payload)
    return ActionResponse(ok=True, message="konfigurasi Auto Trade berhasil disimpan", detail=values)


@router.get("/scalping-setup", tags=["monitor"])
def scalping_setup(symbol: str | None = None) -> dict:
    if symbol is None:
        return {"symbols": get_all_scalping_setups()}
    try:
        return get_scalping_setup(symbol)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/scalping-setup", response_model=ActionResponse, tags=["control"])
def update_scalping_setup(req: ScalpingSetupRequest) -> ActionResponse:
    payload = req.model_dump()
    symbol = payload.pop("symbol")
    values = save_scalping_setup(symbol, payload)
    return ActionResponse(ok=True, message=f"konfigurasi Counter Scalping M1 {symbol} tersimpan", detail={"symbol": symbol, **values})


@router.get("/account", response_model=AccountResponse, tags=["monitor"])
def account() -> AccountResponse:
    info = connection.account_info()
    return AccountResponse(connected=info is not None, info=info)


@router.get("/accounts", tags=["monitor"])
def accounts() -> dict:
    """List the primary account and isolated account workers."""
    return {"accounts": account_manager.list() if not is_account_worker() else []}


@router.delete("/accounts/{account_id}", response_model=ActionResponse, tags=["control"])
def remove_account(account_id: str) -> ActionResponse:
    if account_id == "default":
        return ActionResponse(ok=False, message="akun utama tidak dapat dihapus")
    removed = account_manager.remove(account_id)
    return ActionResponse(ok=removed, message="worker akun dihentikan" if removed else "akun tidak ditemukan")


@router.post("/account/login", response_model=ActionResponse, tags=["control"])
def account_login(req: MT5LoginRequest) -> ActionResponse:
    if not is_account_worker() and req.account_id and req.account_id != "default":
        try:
            password = req.password or account_manager.saved_password(req.account_id)
            if not password:
                return ActionResponse(ok=False, message="password wajib diisi atau pilih akun dengan password tersimpan")
            worker = account_manager.add(
                req.account_id, req.label or req.account_id, req.login, password,
                req.server.strip(), req.terminal_path or "",
            )
            return ActionResponse(ok=True, message=f"akun {worker.label} berhasil ditambahkan", detail={"account_id": worker.account_id})
        except (ValueError, RuntimeError) as exc:
            return ActionResponse(ok=False, message=str(exc))
    if not req.password:
        return ActionResponse(ok=False, message="password akun utama wajib diisi")
    ok, message = connection.login(
        req.login,
        req.password,
        req.server.strip(),
        req.terminal_path,
    )
    return ActionResponse(ok=ok, message=message)


@router.post("/account/logout", response_model=ActionResponse, tags=["control"])
def account_logout() -> ActionResponse:
    bot = _get_bot()
    if bot.running:
        bot.stop()
    connection.logout()
    return ActionResponse(ok=True, message="akun MT5 berhasil logout dan auto trade dihentikan")


@router.get("/positions", response_model=PositionsResponse, tags=["monitor"])
def positions(all_account: bool = True) -> PositionsResponse:
    pos = position_manager.get_open_positions(bot_only=not all_account)
    return PositionsResponse(
        count=len(pos),
        total_profit=round(sum(float(item.get("profit", 0.0)) for item in pos), 2),
        positions=pos,
    )


@router.get("/trade-history", response_model=TradeHistoryResponse, tags=["monitor"])
def trade_history(days: int = 30, limit: int = 100, all_account: bool = True, result: str = "all", date_from: date | None = None, date_to: date | None = None, tz_offset: int = 0, symbol: str | None = None) -> TradeHistoryResponse:
    from app.mt5.trade_history import get_closed_deals, summarize_closed_deals

    days = max(1, min(days, 3650))
    limit = max(1, min(limit, 1000))
    result = result.lower()
    offset = max(-840, min(tz_offset, 840))
    start = datetime.combine(date_from, time.min, tzinfo=timezone.utc) + timedelta(minutes=offset) if date_from else None
    end = datetime.combine(date_to + timedelta(days=1), time.min, tzinfo=timezone.utc) + timedelta(minutes=offset) if date_to else None
    selected_symbol = symbol.strip().upper() if symbol and symbol.strip() else None
    all_deals = get_closed_deals(days=days, limit=None, bot_only=not all_account, date_from=start, date_to=end, symbol=selected_symbol)
    if result in {"win", "loss", "be"}:
        all_deals = [deal for deal in all_deals if str(deal.get("result", "")).lower() == result]
    all_deals.sort(key=lambda deal: (str(deal.get("time", "")), int(deal.get("ticket", 0))), reverse=True)
    return TradeHistoryResponse(
        days=days,
        summary=summarize_closed_deals(all_deals),
        deals=all_deals[:limit],
    )


@router.get("/trade-hour-analysis", tags=["monitor"])
def trade_hour_analysis(days: int = 3650, tz_offset: int = 0, all_account: bool = True, result: str = "all", date_from: date | None = None, date_to: date | None = None, symbol: str | None = None) -> dict:
    from app.mt5.trade_history import get_closed_deals, summarize_by_open_hour

    offset = max(-840, min(tz_offset, 840))
    start = datetime.combine(date_from, time.min, tzinfo=timezone.utc) + timedelta(minutes=offset) if date_from else None
    end = datetime.combine(date_to + timedelta(days=1), time.min, tzinfo=timezone.utc) + timedelta(minutes=offset) if date_to else None
    selected_symbol = symbol.strip().upper() if symbol and symbol.strip() else None
    deals = get_closed_deals(days=max(1, min(days, 3650)), limit=None, bot_only=not all_account, date_from=start, date_to=end, symbol=selected_symbol)
    result = result.lower()
    if result in {"win", "loss", "be"}:
        deals = [deal for deal in deals if str(deal.get("result", "")).lower() == result]
    return summarize_by_open_hour(deals, offset)


@router.get("/capital-curve", tags=["monitor"])
def capital_curve(days: int = 3650) -> dict:
    from app.mt5.trade_history import get_capital_curve

    return get_capital_curve(days=max(1, min(days, 3650)))


@router.get("/activity-logs", tags=["monitor"])
def activity_logs(limit: int = 300) -> dict:
    """Return recent application events for the dashboard, newest first."""
    limit = max(1, min(limit, 1000))
    pattern = re.compile(
        r"^(?P<time>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \| "
        r"(?P<level>\w+)\s*\| (?P<source>.*?) - (?P<message>.*)$"
    )
    rows: list[dict[str, str]] = []
    files = sorted(Path(os.getenv("LOG_DIR", "logs")).glob("trading_*.log"), key=lambda path: path.stat().st_mtime, reverse=True)
    for path in files:
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in reversed(lines):
            match = pattern.match(line)
            if not match:
                continue
            row = match.groupdict()
            row["level"] = row["level"].strip().upper()
            rows.append(row)
            if len(rows) >= limit:
                return {"count": len(rows), "logs": rows}
    return {"count": len(rows), "logs": rows}


@router.get("/signal", response_model=SignalResponse, tags=["monitor"])
def signal(timeframe: str | None = None, symbol: str | None = None) -> SignalResponse:
    bot = _get_bot()
    tf = timeframe or bot.PRIMARY_TF
    selected_symbol = symbol.strip() if symbol else settings.symbol
    if symbol and not connection.symbol_info(selected_symbol):
        raise HTTPException(status_code=400, detail=f"symbol {selected_symbol} tidak tersedia di MT5")
    sig = bot.compute_signal_now(tf, selected_symbol)
    payload = sig.to_dict()
    confidence = float(payload.get("confidence", 0.0) or 0.0)
    ml = payload.get("ml", {})
    plan = bot.preview_trade_plan(sig, symbol=selected_symbol)
    threshold = float(get_trading_setup()["confidence_threshold"])
    if confidence >= threshold and isinstance(ml, dict):
        recommendation = "BUY" if float(ml.get("buy", 0.0)) >= float(ml.get("sell", 0.0)) else "SELL"
        sr_ok, sr_reason = bot.confidence_support_resistance_check(sig, recommendation)
        plan = bot.preview_trade_plan(
            sig,
            direction_override=recommendation,
            atr_mult=0.60,
            risk_reward=1.0,
            use_levels=False,
            symbol=selected_symbol,
        )
        if plan is not None:
            plan["recommendation"] = recommendation
            plan["confidence"] = round(confidence, 4)
            model_ready = bool(sig.ml.get("model"))
            blocked_reasons = []
            if not model_ready:
                blocked_reasons.append("model ML belum tersedia")
            if not sr_ok:
                blocked_reasons.append(sr_reason)
            plan["execution_allowed"] = model_ready and sr_ok
            plan["status"] = "READY" if plan["execution_allowed"] else "ANALYSIS_ONLY"
            plan["blocked_reasons"] = blocked_reasons
    open_positions = position_manager.get_open_positions(selected_symbol, bot_only=False)
    position_rows = [
        {
            "ticket": position.get("ticket"),
            "direction": position.get("type_str"),
            "lot": position.get("volume"),
            "entry": position.get("price_open"),
            "stop_loss": position.get("sl"),
            "take_profit": position.get("tp"),
            "floating_profit": position.get("profit"),
        }
        for position in open_positions
    ]
    if plan is None and position_rows:
        plan = {"status": "MANAGE_OPEN_POSITION"}
    if plan is not None:
        direction = plan.get("direction")
        existing_directions = {position.get("direction") for position in position_rows}
        if not position_rows:
            position_mode = "NEW_POSITION"
        elif direction in existing_directions:
            position_mode = "ADD_POSITION"
        elif direction:
            position_mode = "OPPOSITE_POSITION_OPEN"
        else:
            position_mode = "MANAGE_POSITION"
        plan["position_mode"] = position_mode
        plan["open_position_count"] = len(position_rows)
        plan["open_floating_profit"] = round(sum(float(p.get("floating_profit") or 0.0) for p in position_rows), 2)
        plan["open_positions"] = position_rows
    return SignalResponse(
        timeframe=tf,
        signal=payload,
        trade_plan=plan,
    )


@router.post("/trade/start", response_model=ActionResponse, tags=["control"])
def trade_start() -> ActionResponse:
    bot = _get_bot()
    started = bot.start()
    if started:
        message = "bot started"
    elif bot.running:
        message = "bot already running"
    else:
        message = "bot not started; check MT5 installation, path, login, and server"
    return ActionResponse(ok=started, message=message)


@router.post("/trade/stop", response_model=ActionResponse, tags=["control"])
def trade_stop() -> ActionResponse:
    bot = _get_bot()
    stopped = bot.stop()
    return ActionResponse(ok=stopped, message="bot stopped" if stopped else "bot not running")


@router.post("/trade/confidence-auto/start", response_model=ActionResponse, tags=["control"])
def confidence_auto_start() -> ActionResponse:
    bot = _get_bot()
    started = bot.start(confidence_auto=True)
    return ActionResponse(
        ok=started,
        message=f"auto trade {bot.active_strategy} aktif" if started else f"gagal mengaktifkan auto trade {bot.active_strategy}",
        detail={"strategy": bot.active_strategy, "threshold": get_trading_setup()["confidence_threshold"], "symbols": bot.auto_symbols},
    )


@router.post("/trade/strategy/{strategy_id}", response_model=ActionResponse, tags=["control"])
def select_strategy(strategy_id: str) -> ActionResponse:
    bot = _get_bot()
    if strategy_id not in {"confidence_m5", "recovery_m1"}:
        return ActionResponse(ok=False, message="strategi tidak dikenal")
    if bot.running:
        return ActionResponse(ok=False, message="hentikan Auto Trade sebelum mengganti strategi")
    setup = get_trading_setup()
    setup["active_strategy"] = strategy_id
    saved = save_trading_setup(setup)
    label = "Confidence M5" if strategy_id == "confidence_m5" else "Counter Basket Scalping M1"
    return ActionResponse(ok=True, message=f"strategi aktif: {label}", detail={"active_strategy": saved["active_strategy"]})


@router.post("/trade/confidence-auto/markets", response_model=ActionResponse, tags=["control"])
def confidence_auto_markets(req: AutoMarketsRequest) -> ActionResponse:
    bot = _get_bot()
    symbols = list(dict.fromkeys(symbol.strip() for symbol in req.symbols if symbol.strip()))
    unavailable = [symbol for symbol in symbols if not connection.symbol_info(symbol)]
    if unavailable:
        return ActionResponse(ok=False, message=f"symbol tidak tersedia di MT5: {', '.join(unavailable)}")
    try:
        selected = bot.set_auto_symbols(symbols)
    except ValueError as exc:
        return ActionResponse(ok=False, message=str(exc))
    return ActionResponse(
        ok=True,
        message=f"market Auto Confidence: {' + '.join(selected)}",
        detail={"symbols": selected, "running": bot.confidence_auto},
    )


@router.post("/trade/confidence-auto/stop", response_model=ActionResponse, tags=["control"])
def confidence_auto_stop() -> ActionResponse:
    bot = _get_bot()
    stopped = bot.stop()
    bot.confidence_auto = False
    return ActionResponse(ok=True, message="auto trade confidence berhenti", detail={"bot_stopped": stopped})


@router.post("/trade/close-all", response_model=ActionResponse, tags=["control"])
def trade_close_all() -> ActionResponse:
    results = order_executor.close_all_positions(bot_only=False, all_symbols=True)
    if not results:
        return ActionResponse(ok=True, message="tidak ada posisi akun yang terbuka", detail=[])
    succeeded = sum(bool(result.get("ok")) for result in results)
    failed = len(results) - succeeded
    return ActionResponse(
        ok=failed == 0,
        message=f"tutup semua: {succeeded} berhasil, {failed} gagal",
        detail=results,
    )


@router.post("/trade/manual", response_model=ActionResponse, tags=["control"])
def trade_manual(req: ManualTradeRequest) -> ActionResponse:
    """Place a confirmed manual dashboard order with a fresh compact plan."""
    bot = _get_bot()
    minimum_lot = order_executor.configured_minimum_lot(settings.symbol)
    if req.lot < minimum_lot:
        return ActionResponse(
            ok=False,
            message=f"lot minimum {settings.symbol} adalah {minimum_lot:.2f}",
            detail={"symbol": settings.symbol, "minimum_lot": minimum_lot},
        )
    sig = bot.compute_signal_now(bot.PRIMARY_TF)
    plan = bot.preview_trade_plan(
        sig,
        direction_override=req.direction,
        atr_mult=0.60,
        risk_reward=1.0,
        use_levels=False,
    )
    if plan is None:
        return ActionResponse(ok=False, message="gagal membuat rencana order dari data market terbaru")
    if req.direction == "BUY":
        result = order_executor.open_buy(
            settings.symbol, req.lot, float(plan["stop_loss"]), float(plan["take_profit"]),
            confidence_comment(sig.confidence, req.direction), enforce_spread=False,
        )
    else:
        result = order_executor.open_sell(
            settings.symbol, req.lot, float(plan["stop_loss"]), float(plan["take_profit"]),
            confidence_comment(sig.confidence, req.direction), enforce_spread=False,
        )
    detail = {"plan": plan, "result": result.to_dict()}
    return ActionResponse(ok=result.ok, message=result.message, detail=detail)


@router.post("/trade/set-level", response_model=ActionResponse, tags=["control"])
def trade_set_level(req: BulkLevelRequest) -> ActionResponse:
    results = order_executor.set_all_position_level(req.level, req.price, settings.symbol)
    if not results:
        return ActionResponse(ok=False, message=f"tidak ada posisi {settings.symbol} yang terbuka", detail=[])
    succeeded = sum(bool(result.get("ok")) for result in results)
    failed = len(results) - succeeded
    return ActionResponse(
        ok=failed == 0,
        message=f"set {req.level} {settings.symbol}: {succeeded} berhasil, {failed} gagal",
        detail=results,
    )


@router.post("/train", response_model=ActionResponse, tags=["ml"])
def train(req: TrainRequest) -> ActionResponse:
    if not os.path.exists(req.csv):
        raise HTTPException(status_code=400, detail=f"CSV not found: {req.csv}")
    try:
        from app.ml.train_xgboost import train as train_model

        metrics = train_model(
            csv_path=req.csv,
            horizon=req.horizon,
            atr_mult=req.atr_mult,
            test_size=req.test_size,
        )
        return ActionResponse(ok=True, message="training complete", detail=metrics)
    except Exception as exc:
        logger.exception("Training failed: {}", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/data/export", response_model=ActionResponse, tags=["data"])
def export_data(req: ExportRequest) -> ActionResponse:
    """Export broker candle history to ``data/<symbol>_<timeframe>.csv``."""
    try:
        from app.mt5.market_data import export_candles_csv

        detail = export_candles_csv(
            symbol=req.symbol,
            timeframe=req.timeframe,
            count=req.count,
        )
        return ActionResponse(ok=True, message="candle export complete", detail=detail)
    except Exception as exc:
        logger.exception("Candle export failed: {}", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/backtest", response_model=ActionResponse, tags=["ml"])
def backtest(req: BacktestRequest) -> ActionResponse:
    if not os.path.exists(req.csv):
        raise HTTPException(status_code=400, detail=f"CSV not found: {req.csv}")
    try:
        from app.backtest.backtester import run_backtest
        from app.backtest.report import summarize
        from app.mt5.market_data import load_candles_csv

        df = load_candles_csv(req.csv)
        result = run_backtest(
            df,
            start_balance=req.start_balance,
            warmup=req.warmup,
            max_hold=req.max_hold,
            signal_lookback=req.signal_lookback,
            account_profile=req.account_profile,
            use_historical_spread=req.use_historical_spread,
            commission_per_lot_side=req.commission_per_lot_side,
            slippage_points=req.slippage_points,
        )
        stats = summarize(result)
        return ActionResponse(ok=True, message="backtest complete", detail=stats)
    except Exception as exc:
        logger.exception("Backtest failed: {}", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# --- Minimal embedded dashboard (no external assets / CDN) -----------------
_DASHBOARD_HTML = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>AI Trading Agent</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 0; background:#0f1115; color:#e6e6e6; }
  header { padding: 16px 24px; background:#171a21; border-bottom:1px solid #262b36; }
  h1 { font-size: 18px; margin:0; }
  main { padding: 24px; display:grid; gap:16px; grid-template-columns: repeat(auto-fit,minmax(280px,1fr)); }
  .card { background:#171a21; border:1px solid #262b36; border-radius:10px; padding:16px; }
  .card h2 { font-size:13px; text-transform:uppercase; letter-spacing:.05em; color:#8a93a6; margin:0 0 12px; }
  pre { white-space:pre-wrap; word-break:break-word; font-size:12px; margin:0; color:#cdd6e4; }
  .row { display:flex; gap:8px; flex-wrap:wrap; margin-bottom:12px; }
  button { background:#2563eb; color:#fff; border:0; padding:8px 14px; border-radius:8px; cursor:pointer; font-size:13px; }
  button.danger { background:#dc2626; } button.gray { background:#374151; }
  .pill { display:inline-block; padding:2px 8px; border-radius:999px; font-size:11px; }
  .on { background:#064e3b; color:#6ee7b7; } .off { background:#3f1d1d; color:#fca5a5; }
</style>
</head>
<body>
<header><h1>🤖 AI Trading Agent <span id="mode" class="pill off">loading…</span></h1></header>
<main>
  <div class="card" style="grid-column:1/-1">
    <h2>Controls</h2>
    <div class="row">
      <button onclick="call('/trade/start','POST')">Start</button>
      <button class="gray" onclick="call('/trade/stop','POST')">Stop</button>
      <button class="danger" onclick="call('/trade/close-all','POST')">Close All</button>
      <button class="gray" onclick="refresh()">Refresh</button>
    </div>
    <pre id="action"></pre>
  </div>
  <div class="card"><h2>Status</h2><pre id="status"></pre></div>
  <div class="card"><h2>Balance &amp; Bot Profit</h2><pre id="summary"></pre></div>
  <div class="card"><h2>Account</h2><pre id="account"></pre></div>
  <div class="card"><h2>Signal (M5)</h2><pre id="signal"></pre></div>
  <div class="card"><h2>Positions</h2><pre id="positions"></pre></div>
</main>
<script>
async function get(p){ const r=await fetch(p); return r.json(); }
async function call(p,m){ const r=await fetch(p,{method:m}); document.getElementById('action').textContent=JSON.stringify(await r.json(),null,2); refresh(); }
function show(id,obj){ document.getElementById(id).textContent=JSON.stringify(obj,null,2); }
async function refresh(){
  try{
    const s=await get('/status'); show('status',s);
    show('summary', {
      currency: s.account_currency,
      balance: s.account_balance,
      equity: s.account_equity,
      floating_profit: s.total_profit,
      bot_positions: s.open_positions
    });
    const m=document.getElementById('mode');
    m.textContent = s.trading_enabled ? 'LIVE' : 'SAFE MODE';
    m.className = 'pill ' + (s.trading_enabled ? 'on' : 'off');
    show('account', await get('/account'));
    show('signal', await get('/signal'));
    show('positions', await get('/positions'));
  }catch(e){ show('status', {error:String(e)}); }
}
refresh(); setInterval(refresh, 10000);
</script>
</body>
</html>
"""
