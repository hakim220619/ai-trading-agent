"""CLI for the standalone universal day-trade strategy backtest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.backtest.day_trade_backtester import run_day_trade_backtest, summarize_day_trade
from app.config import settings
from app.mt5.connection import connection
from app.mt5.market_data import load_candles_csv


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest universal day-trade strategy")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--start-balance", type=float, default=100.0)
    parser.add_argument("--daily-target", type=float, default=30.0)
    parser.add_argument("--daily-loss-limit", type=float, default=20.0)
    parser.add_argument("--risk-percent", type=float, default=1.0)
    parser.add_argument("--max-trades-per-day", type=int, default=3)
    parser.add_argument("--max-hold-bars", type=int, default=24)
    parser.add_argument("--cooldown-bars", type=int, default=6)
    parser.add_argument("--max-spread-points", type=float, default=1500.0)
    parser.add_argument("--commission-per-lot-side", type=float, default=0.0)
    parser.add_argument("--slippage-points", type=float, default=0.0)
    parser.add_argument("--account-profile", default="exness-pro")
    parser.add_argument("--market-profile", choices=["auto", "crypto", "metals", "standard"], default="auto")
    parser.add_argument("--last", type=int, default=0)
    parser.add_argument("--output", default=None, help="Optional JSON report path")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    try:
        settings.symbol = args.symbol.upper()
        candles = load_candles_csv(args.csv)
        if args.last > 0:
            candles = candles.tail(args.last).reset_index(drop=True)
        result = run_day_trade_backtest(
            candles,
            start_balance=args.start_balance,
            daily_profit_target=args.daily_target,
            daily_loss_limit=args.daily_loss_limit,
            risk_percent=args.risk_percent,
            max_trades_per_day=args.max_trades_per_day,
            max_hold_bars=args.max_hold_bars,
            cooldown_bars=args.cooldown_bars,
            max_spread_points=args.max_spread_points,
            commission_per_lot_side=args.commission_per_lot_side,
            slippage_points=args.slippage_points,
            account_profile=args.account_profile,
            market_profile=args.market_profile,
        )
        report = summarize_day_trade(result)
        print("\n===== UNIVERSAL DAY-TRADE BACKTEST =====")
        for key, value in report.items():
            print(f"{key:>24}: {value}")
        print("========================================\n")
        if args.output:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"Report saved to {output}")
    finally:
        connection.disconnect()
