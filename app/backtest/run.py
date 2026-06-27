"""Command-line runner for an OHLCV CSV backtest.

Example:
    python -m app.backtest.run --csv data/BTCUSD_M5.csv --last 2000
"""
from __future__ import annotations

import argparse

from app.backtest.backtester import run_backtest
from app.backtest.report import print_report
from app.config import settings
from app.mt5.connection import connection
from app.mt5.market_data import load_candles_csv


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a trading strategy backtest")
    parser.add_argument("--csv", required=True, help="OHLCV candle CSV path")
    parser.add_argument("--last", type=int, default=0, help="Use only the latest N candles")
    parser.add_argument("--start-balance", type=float, default=1000.0)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--max-hold", type=int, default=96)
    parser.add_argument("--signal-lookback", type=int, default=500)
    parser.add_argument("--symbol", default=None, help="Broker symbol used for contract/PnL math")
    parser.add_argument("--model", default=None, help="XGBoost model path for this dataset")
    parser.add_argument("--account-profile", default="exness-pro")
    parser.add_argument("--commission-per-lot-side", type=float, default=0.0)
    parser.add_argument("--slippage-points", type=float, default=0.0)
    parser.add_argument("--ignore-historical-spread", action="store_true")
    parser.add_argument("--max-spread-points", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    try:
        if args.symbol:
            settings.symbol = args.symbol.upper()
        if args.model:
            settings.model_path = args.model
        if args.max_spread_points is not None:
            settings.max_spread_points = args.max_spread_points
        candles = load_candles_csv(args.csv)
        if args.last > 0:
            candles = candles.tail(args.last).reset_index(drop=True)
        result = run_backtest(
            candles,
            start_balance=args.start_balance,
            warmup=args.warmup,
            max_hold=args.max_hold,
            signal_lookback=args.signal_lookback,
            account_profile=args.account_profile,
            use_historical_spread=not args.ignore_historical_spread,
            commission_per_lot_side=args.commission_per_lot_side,
            slippage_points=args.slippage_points,
        )
        print_report(result)
    finally:
        connection.disconnect()
