"""FastAPI routes for monitoring and control."""
from __future__ import annotations

import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

from app.api.schemas import (
    AccountResponse,
    ActionResponse,
    BacktestRequest,
    PositionsResponse,
    SignalResponse,
    StatusResponse,
    TrainRequest,
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
    """Minimal single-page monitoring dashboard."""
    return _DASHBOARD_HTML


@router.get("/status", response_model=StatusResponse, tags=["monitor"])
def status() -> StatusResponse:
    bot = _get_bot()
    connected = MT5_AVAILABLE and connection.connected
    return StatusResponse(
        running=bot.running,
        trading_enabled=settings.trading_enabled,
        mt5_connected=connected,
        symbol=settings.symbol,
        timeframes=settings.timeframes,
        open_positions=order_executor.count_open_positions(),
        model_loaded=get_model() is not None,
    )


@router.get("/account", response_model=AccountResponse, tags=["monitor"])
def account() -> AccountResponse:
    info = connection.account_info()
    return AccountResponse(connected=info is not None, info=info)


@router.get("/positions", response_model=PositionsResponse, tags=["monitor"])
def positions() -> PositionsResponse:
    pos = position_manager.get_open_positions()
    return PositionsResponse(
        count=len(pos),
        total_profit=round(position_manager.total_profit(), 2),
        positions=pos,
    )


@router.get("/signal", response_model=SignalResponse, tags=["monitor"])
def signal(timeframe: str | None = None) -> SignalResponse:
    bot = _get_bot()
    tf = timeframe or bot.PRIMARY_TF
    sig = bot.compute_signal_now(tf)
    return SignalResponse(timeframe=tf, signal=sig.to_dict())


@router.post("/trade/start", response_model=ActionResponse, tags=["control"])
def trade_start() -> ActionResponse:
    bot = _get_bot()
    started = bot.start()
    return ActionResponse(
        ok=started,
        message="bot started" if started else "bot already running",
    )


@router.post("/trade/stop", response_model=ActionResponse, tags=["control"])
def trade_stop() -> ActionResponse:
    bot = _get_bot()
    stopped = bot.stop()
    return ActionResponse(ok=stopped, message="bot stopped" if stopped else "bot not running")


@router.post("/trade/close-all", response_model=ActionResponse, tags=["control"])
def trade_close_all() -> ActionResponse:
    results = order_executor.close_all_positions()
    return ActionResponse(ok=True, message="close-all executed", detail=results)


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
