"""Export historical MetaTrader 5 candles to CSV.

Example:
    python -m app.mt5.export_data --symbol XAUUSD --timeframe M5 --count 10000
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone

from app.mt5.connection import connection
from app.mt5.market_data import export_candles_csv


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export MT5 candles to CSV")
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--timeframe", default="M5")
    parser.add_argument("--count", type=int, default=10_000)
    parser.add_argument("--output", default=None)
    parser.add_argument("--years", type=float, default=0, help="Export the latest N years")
    parser.add_argument("--from-date", default=None, help="UTC start date, e.g. 2021-06-27")
    parser.add_argument("--to-date", default=None, help="UTC end date; defaults to now")
    return parser.parse_args()


def _utc_date(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


if __name__ == "__main__":
    args = _parse_args()
    try:
        date_to = _utc_date(args.to_date) if args.to_date else datetime.now(timezone.utc)
        date_from = None
        if args.years > 0:
            date_from = date_to - timedelta(days=365.25 * args.years)
        elif args.from_date:
            date_from = _utc_date(args.from_date)
        result = export_candles_csv(
            symbol=args.symbol,
            timeframe=args.timeframe,
            count=args.count,
            path=args.output,
            date_from=date_from,
            date_to=date_to if date_from else None,
        )
        print(result)
        if date_from and result.get("first_time"):
            actual = datetime.fromisoformat(str(result["first_time"])).replace(tzinfo=timezone.utc)
            if actual > date_from + timedelta(days=2):
                print(
                    "WARNING: broker/terminal returned less history than requested. "
                    "Increase MT5 Tools > Options > Charts > Max bars in chart, "
                    "restart MT5, and run the export again."
                )
    finally:
        connection.disconnect()
