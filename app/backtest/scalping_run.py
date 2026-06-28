"""CLI backtester for the separate M1 and M5 scalping strategies."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.backtest.day_trade_backtester import run_day_trade_backtest, summarize_day_trade
from app.config import settings
from app.mt5.connection import connection
from app.mt5.market_data import load_candles_csv


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest M1 or M5 scalping strategy")
    parser.add_argument("--strategy", choices=["m1", "ma-m1", "snd-m1", "m5"], required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--start-balance", type=float, default=1000.0)
    parser.add_argument("--risk-percent", type=float, default=0.5)
    parser.add_argument("--fixed-lot", type=float, default=None, help="Use a fixed lot instead of risk-based sizing")
    parser.add_argument("--weekdays-only", action="store_true", help="Allow entries Monday through Friday only")
    parser.add_argument("--session-start", type=int, default=None, help="Daily start hour in candle/server time (0-23)")
    parser.add_argument("--session-hours", type=int, default=24, help="Maximum daily running window")
    parser.add_argument("--m1-min-setups", type=int, choices=[1, 2, 3, 4], default=3, help="M1 setup confirmations; 2 is active, 3 is selective")
    parser.add_argument("--max-consecutive-losses", type=int, default=0, help="Stop entries for the day after this many losses; 0 disables")
    parser.add_argument("--max-trades-per-day", type=int, default=12)
    parser.add_argument("--max-hold-bars", type=int, default=None)
    parser.add_argument("--cooldown-bars", type=int, default=None)
    parser.add_argument("--daily-target", type=float, default=1000000.0)
    parser.add_argument("--daily-loss-limit", type=float, default=20.0)
    parser.add_argument("--max-spread-points", type=float, default=300.0)
    parser.add_argument("--commission-per-lot-side", type=float, default=0.0)
    parser.add_argument("--slippage-points", type=float, default=0.0)
    parser.add_argument("--last", type=int, default=0)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    settings.symbol = args.symbol.upper()
    candles = load_candles_csv(args.csv)
    if args.last > 0:
        candles = candles.tail(args.last).reset_index(drop=True)
    max_hold = args.max_hold_bars or (15 if args.strategy == "m1" else 12)
    cooldown = args.cooldown_bars if args.cooldown_bars is not None else (3 if args.strategy == "m1" else 2)
    result = run_day_trade_backtest(
        candles,
        start_balance=args.start_balance,
        daily_profit_target=args.daily_target,
        daily_loss_limit=args.daily_loss_limit,
        risk_percent=args.risk_percent,
        max_trades_per_day=args.max_trades_per_day,
        max_hold_bars=max_hold,
        cooldown_bars=cooldown,
        max_spread_points=args.max_spread_points,
        commission_per_lot_side=args.commission_per_lot_side,
        slippage_points=args.slippage_points,
        strategy=args.strategy,
        fixed_lot=args.fixed_lot,
        weekdays_only=args.weekdays_only,
        session_start_hour=args.session_start,
        session_hours=args.session_hours,
        m1_min_setups=args.m1_min_setups,
        max_consecutive_losses=args.max_consecutive_losses,
    )
    report = summarize_day_trade(result)
    print(f"\n===== SCALPING {args.strategy.upper()} BACKTEST =====")
    for key, value in report.items():
        print(f"{key:>24}: {value}")
    print("================================\n")
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Report saved to {output}")


if __name__ == "__main__":
    try:
        main()
    finally:
        connection.disconnect()
