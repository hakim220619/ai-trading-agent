"""FastAPI routes for monitoring and control."""
from __future__ import annotations

import os
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
    SignalResponse,
    SymbolRequest,
    StatusResponse,
    TrainRequest,
    TradeHistoryResponse,
)
from app.config import settings
from app.mt5 import order_executor, position_manager
from app.mt5.connection import MT5_AVAILABLE, connection
from app.ml.predict import get_model
from app.utils.logger import logger

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
    return {
        "active_symbol": settings.symbol,
        "auto_symbols": _get_bot().auto_symbols,
        "active_timeframe": "M5",
        "confidence_auto": _get_bot().confidence_auto,
        "live": [
            {"name": "Technical Trend", "detail": "EMA20/EMA50/EMA200 + RSI"},
            {"name": "Support & Resistance", "detail": "Swing, breakout, retest, BOS, dan CHoCH"},
            {"name": "XGBoost Confidence", "detail": "BUY/SELL langsung dari probabilitas terbesar jika confidence >= 65%; tanpa konfirmasi teknis/MTF/spread"},
            {"name": "Multi-Timeframe", "detail": "Konfirmasi konteks M15 dan H1"},
            {"name": "Risk & Execution", "detail": "ATR SL/TP adaptif, maksimal 3 posisi, trailing SL mengunci profit setiap kenaikan $1"},
        ],
        "backtest_only": [
            {"name": "Combined Scalping M1", "module": "scalping_m1_strategy"},
            {"name": "MA Crossover M1", "module": "scalping_ma_m1_strategy"},
            {"name": "Supply & Demand M1", "module": "scalping_snd_m1_strategy"},
            {"name": "Combined Scalping M5", "module": "scalping_m5_strategy"},
            {"name": "Universal Day Trade", "module": "day_trade_strategy"},
        ],
    }


@router.get("/status", response_model=StatusResponse, tags=["monitor"])
def status() -> StatusResponse:
    bot = _get_bot()
    connected = MT5_AVAILABLE and connection.ensure_connected()
    account_info = connection.account_info() if connected else None
    positions = position_manager.get_open_positions(bot_only=False) if connected else []
    total_profit = round(sum(float(p.get("profit", 0.0)) for p in positions), 2)
    trade_status = connection.trading_status() if connected else {
        "terminal_trade_allowed": False,
        "account_trade_allowed": False,
        "trade_api_disabled": True,
    }
    return StatusResponse(
        running=bot.running,
        trading_enabled=settings.trading_enabled,
        mt5_connected=connected,
        symbol=settings.symbol,
        timeframes=settings.timeframes,
        open_positions=len(positions),
        model_loaded=get_model() is not None,
        account_balance=(round(float(account_info["balance"]), 2) if account_info else None),
        account_equity=(round(float(account_info["equity"]), 2) if account_info else None),
        account_currency=(str(account_info["currency"]) if account_info else None),
        total_profit=total_profit,
        confidence_auto=bot.confidence_auto,
        auto_symbols=bot.auto_symbols,
        max_open_positions=settings.max_open_positions,
        **trade_status,
    )


@router.post("/symbol", response_model=ActionResponse, tags=["control"])
def select_symbol(req: SymbolRequest) -> ActionResponse:
    bot = _get_bot()
    if not connection.symbol_info(req.symbol):
        return ActionResponse(ok=False, message=f"symbol {req.symbol} tidak tersedia di MT5")
    bot.set_symbol(req.symbol)
    return ActionResponse(
        ok=True,
        message=f"market aktif diubah ke {req.symbol}",
        detail={"symbol": req.symbol, "bot_running": bot.running},
    )


@router.get("/account", response_model=AccountResponse, tags=["monitor"])
def account() -> AccountResponse:
    info = connection.account_info()
    return AccountResponse(connected=info is not None, info=info)


@router.post("/account/login", response_model=ActionResponse, tags=["control"])
def account_login(req: MT5LoginRequest) -> ActionResponse:
    ok, message = connection.login(req.login, req.password, req.server.strip())
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
def trade_history(days: int = 30, limit: int = 100, all_account: bool = True) -> TradeHistoryResponse:
    from app.mt5.trade_history import get_closed_deals, summarize_closed_deals

    days = max(1, min(days, 3650))
    limit = max(1, min(limit, 1000))
    all_deals = get_closed_deals(days=days, limit=None, bot_only=not all_account)
    all_deals.sort(key=lambda deal: (str(deal.get("time", "")), int(deal.get("ticket", 0))), reverse=True)
    return TradeHistoryResponse(
        days=days,
        summary=summarize_closed_deals(all_deals),
        deals=all_deals[:limit],
    )


@router.get("/capital-curve", tags=["monitor"])
def capital_curve(days: int = 3650) -> dict:
    from app.mt5.trade_history import get_capital_curve

    return get_capital_curve(days=max(1, min(days, 3650)))


@router.get("/signal", response_model=SignalResponse, tags=["monitor"])
def signal(timeframe: str | None = None) -> SignalResponse:
    bot = _get_bot()
    tf = timeframe or bot.PRIMARY_TF
    sig = bot.compute_signal_now(tf)
    payload = sig.to_dict()
    confidence = float(payload.get("confidence", 0.0) or 0.0)
    ml = payload.get("ml", {})
    plan = bot.preview_trade_plan(sig)
    if confidence >= 0.65 and isinstance(ml, dict):
        recommendation = "BUY" if float(ml.get("buy", 0.0)) >= float(ml.get("sell", 0.0)) else "SELL"
        plan = bot.preview_trade_plan(
            sig,
            direction_override=recommendation,
            atr_mult=0.60,
            risk_reward=1.0,
            use_levels=False,
        )
        if plan is not None:
            plan["recommendation"] = recommendation
            plan["confidence"] = round(confidence, 4)
            plan["execution_allowed"] = sig.action == recommendation
            plan["status"] = "READY" if sig.action == recommendation else "ANALYSIS_ONLY"
            plan["blocked_reasons"] = [] if sig.action == recommendation else sig.reasons
    open_positions = position_manager.get_open_positions(settings.symbol, bot_only=False)
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
        message="auto trade confidence aktif" if started else "gagal mengaktifkan auto trade confidence",
        detail={"threshold": 0.65, "symbols": bot.auto_symbols},
    )


@router.post("/trade/confidence-auto/markets", response_model=ActionResponse, tags=["control"])
def confidence_auto_markets(req: AutoMarketsRequest) -> ActionResponse:
    bot = _get_bot()
    unavailable = [symbol for symbol in req.symbols if not connection.symbol_info(symbol)]
    if unavailable:
        return ActionResponse(ok=False, message=f"symbol tidak tersedia di MT5: {', '.join(unavailable)}")
    selected = bot.set_auto_symbols(req.symbols)
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
            "dashboard-buy", enforce_spread=False,
        )
    else:
        result = order_executor.open_sell(
            settings.symbol, req.lot, float(plan["stop_loss"]), float(plan["take_profit"]),
            "dashboard-sell", enforce_spread=False,
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
