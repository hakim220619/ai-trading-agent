# 🤖 AI Trading Agent — Forex / XAUUSD (Python + MT5 + XGBoost)

An automated trading agent for **XAUUSD / Forex** that connects to **MetaTrader 5**,
analyses the market across multiple timeframes (M1, M5, M15, H1), combines a
rule-based strategy with an **XGBoost** machine-learning model, and manages risk,
orders and positions automatically. It ships with **backtesting**, a **training
pipeline**, and a lightweight **FastAPI dashboard**.

> ⚠️ **Trading is risky.** This software can lose money. It defaults to a **SAFE
> mode** (`TRADING_ENABLED=false`) that computes and logs signals but never sends
> a live order. Read the [Risk Disclaimer](#-risk-disclaimer) before going live.

---

## ✨ Features

| Area | What it does |
|------|--------------|
| **MT5 connection** | Connect/login, account & symbol info, spread check, **auto-reconnect** |
| **Market data** | OHLCV candles, multi-timeframe (M1/M5/M15/H1), default 500 candles |
| **Indicators** | EMA 20/50/200, RSI 14, MACD, ATR 14, Bollinger Bands, candle body/wicks, volume avg |
| **Support/Resistance** | Swing highs/lows, nearest support/resistance, breakout & retest detection |
| **Signal engine** | Rule-based BUY/SELL/HOLD **fused** with XGBoost probability + spread guard |
| **ML model** | Feature engineering, training script, persisted model, graceful neutral fallback |
| **Risk manager** | Position sizing by % risk, ATR/swing-based SL, RR-based TP |
| **Order executor** | open_buy/open_sell, close_all, close_profit, trailing stop, full validation |
| **Position manager** | Profit-target close, trailing, reverse-on-opposite-signal |
| **Backtest** | Event-driven backtester + performance report (win rate, PF, drawdown, ROI) |
| **Dashboard** | FastAPI endpoints + a no-dependency HTML monitoring page |
| **Safety** | SAFE mode by default, max positions, spread cap, duplicate-position guard, full decision logging |

---

## 🖥️ Platform note (IMPORTANT)

The `MetaTrader5` Python package **only runs reliably on Windows**.

- **Windows (VPS / VM):** full functionality — live data + live trading.
- **macOS / Linux:** the MT5 import is *guarded*, so you can still **train models**
  and **run backtests** from CSV data. Live data and live trading are disabled.

**Recommended setup:** develop/backtest on your Mac, then deploy and run live on a
**Windows VPS** (or a Windows VM) with MetaTrader 5 installed and logged in.

---

## 📦 Project structure

```
ai-trading-agent/
├── app/
│   ├── main.py                 # TradingBot loop + FastAPI app + entry point
│   ├── config.py               # pydantic-settings config from .env
│   ├── mt5/
│   │   ├── connection.py       # connect, reconnect, account/symbol/spread
│   │   ├── market_data.py      # candle fetching + CSV loader
│   │   ├── order_executor.py   # open/close/trailing with validation
│   │   └── position_manager.py # profit target, trailing, reversal exits
│   ├── strategy/
│   │   ├── indicators.py       # EMA/RSI/MACD/ATR/BB/wicks
│   │   ├── support_resistance.py
│   │   ├── signal_generator.py # rule + ML fusion -> BUY/SELL/HOLD
│   │   └── risk_manager.py     # lot sizing + SL/TP plan
│   ├── ml/
│   │   ├── feature_engineering.py
│   │   ├── train_xgboost.py    # training CLI
│   │   ├── predict.py          # model load + probabilities
│   │   └── models/             # saved model goes here
│   ├── backtest/
│   │   ├── backtester.py
│   │   └── report.py
│   ├── api/
│   │   ├── routes.py           # endpoints + embedded dashboard
│   │   └── schemas.py
│   └── utils/
│       ├── logger.py           # loguru setup
│       └── helpers.py
├── data/                       # put your CSV candle exports here
├── logs/
├── .env.example
├── requirements.txt
└── README.md
```

---

## 🚀 Installation

```bash
# 1. Clone / open the project
cd ai-trading-agent

# 2. Create a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

> On macOS/Linux the `MetaTrader5` line in `requirements.txt` is skipped
> automatically (it is marked `platform_system == "Windows"`).

---

## ⚙️ Setup MetaTrader 5

1. Install **MetaTrader 5** (Windows) and log in to your broker account.
2. In MT5: **Tools → Options → Expert Advisors → Allow Algo Trading**.
3. Make sure the symbol (e.g. `XAUUSD`) is visible in **Market Watch**.
4. Note your **login**, **password**, and **server** name.

---

## 🔐 Configure `.env`

Copy the example and fill in your credentials (never commit `.env`):

```bash
cp .env.example .env
```

```dotenv
MT5_LOGIN=12345678
MT5_PASSWORD=your-password
MT5_SERVER=Your-Broker-Server

SYMBOL=XAUUSD
TIMEFRAMES=M1,M5,M15,H1
CANDLES=500

LOT_DEFAULT=0.01
RISK_PERCENT=1
RISK_REWARD=2.0
MAX_SPREAD_POINTS=300
MAX_OPEN_POSITIONS=2

TRADING_ENABLED=false      # keep false until fully tested!
TARGET_PROFIT_MONEY=5
TRAILING_STOP=true
MAGIC_NUMBER=220619

ML_PROB_THRESHOLD=0.70
MODEL_PATH=app/ml/models/xgboost_model.json
```

---

## 🧠 Train the model

You need historical candles as CSV with columns:
`time, open, high, low, close, volume` (or `tick_volume`), optional `spread`.

You can export this from MT5, or from any data provider.

```bash
python -m app.ml.train_xgboost --csv data/XAUUSD_M5.csv --horizon 1 --atr-mult 0.5
```

This prints **accuracy / precision / recall**, logs top feature importances, and
saves the model to `app/ml/models/xgboost_model.json`.

> If no model is present, the agent still runs — `predict_signal` returns a neutral
> 50/50 and the ML rule simply won't pass, so it behaves conservatively.

---

## 📊 Run a backtest

```bash
# CLI quick check via Python
python -c "from app.mt5.market_data import load_candles_csv; \
from app.backtest.backtester import run_backtest; \
from app.backtest.report import print_report; \
print_report(run_backtest(load_candles_csv('data/XAUUSD_M5.csv')))"
```

…or via the API (see below) using `POST /backtest`.

The report includes: trades, win rate, net profit, ROI, profit factor, average
win/loss, and max drawdown.

---

## 🟢 Run the dashboard / API

```bash
python -m app.main
# or
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000/** for the dashboard.

### Endpoints

| Method | Path | Description |
|-------:|------|-------------|
| GET  | `/` | HTML monitoring dashboard |
| GET  | `/status` | Bot + connection + model status |
| GET  | `/account` | MT5 account info |
| GET  | `/positions` | Open positions + floating P/L |
| GET  | `/signal?timeframe=M5` | Current signal (no trade) |
| POST | `/trade/start` | Start the live decision loop |
| POST | `/trade/stop` | Stop the loop |
| POST | `/trade/close-all` | Close all bot positions |
| POST | `/train` | Train model `{ "csv": "data/XAUUSD_M5.csv" }` |
| POST | `/backtest` | Backtest `{ "csv": "data/XAUUSD_M5.csv" }` |

Interactive docs at **http://localhost:8000/docs**.

---

## ▶️ Go live (only after testing)

1. Run on a **Windows VPS** with MT5 installed & logged in.
2. Backtest and train until you are satisfied with the results.
3. Set `TRADING_ENABLED=true` in `.env`.
4. Start the API, then `POST /trade/start` (or click **Start** on the dashboard).
5. Watch the logs in `logs/` and the dashboard.

The bot only enters when **all** rules align:

**BUY:** EMA20>EMA50 · price>EMA200 · RSI 50–70 · near support · ML buy ≥ threshold · spread OK
**SELL:** EMA20<EMA50 · price<EMA200 · RSI 30–50 · near resistance · ML sell ≥ threshold · spread OK
Otherwise → **HOLD**. Every decision and its reasons are logged.

---

## 🛡️ Safety design

- **SAFE mode by default** — `TRADING_ENABLED=false` blocks every live order.
- Orders are blocked if: market closed, spread too high, duplicate position,
  max positions reached, or invalid lot/SL/TP.
- `MAX_OPEN_POSITIONS` limits concurrent exposure (anti-overtrade).
- All orders are tagged with `MAGIC_NUMBER` so the bot only ever touches its own
  trades.
- Every BUY/SELL/HOLD decision is logged with its full reasoning.

---

## ⚠️ Risk Disclaimer

Trading foreign exchange and gold (XAUUSD) on margin carries a **high level of
risk** and may not be suitable for all investors. Past performance — including
backtest results — is **not indicative of future results**. This project is
provided for **educational purposes** and **without any warranty**. You are solely
responsible for any financial losses. **Never trade with money you cannot afford
to lose.** Always test on a **demo account** first.
