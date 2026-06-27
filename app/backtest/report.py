"""Summarise backtest results into metrics."""
from __future__ import annotations

from typing import Any

from app.backtest.backtester import BacktestResult


def summarize(result: BacktestResult) -> dict[str, Any]:
    """Compute performance statistics from a BacktestResult."""
    trades = result.trades
    n = len(trades)
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]

    gross_profit = sum(t.pnl for t in wins)
    gross_loss = abs(sum(t.pnl for t in losses))
    net = result.end_balance - result.start_balance

    win_rate = (len(wins) / n * 100.0) if n else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
    avg_win = (gross_profit / len(wins)) if wins else 0.0
    avg_loss = (gross_loss / len(losses)) if losses else 0.0

    max_dd = _max_drawdown(result.equity_curve)
    roi = (net / result.start_balance * 100.0) if result.start_balance else 0.0

    return {
        "trades": n,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": round(win_rate, 2),
        "net_profit": round(net, 2),
        "roi_pct": round(roi, 2),
        "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else "inf",
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "start_balance": round(result.start_balance, 2),
        "end_balance": round(result.end_balance, 2),
    }


def _max_drawdown(equity: list[float]) -> float:
    """Maximum peak-to-trough drawdown as a percentage."""
    if not equity:
        return 0.0
    peak = equity[0]
    max_dd = 0.0
    for value in equity:
        peak = max(peak, value)
        if peak > 0:
            dd = (peak - value) / peak * 100.0
            max_dd = max(max_dd, dd)
    return max_dd


def print_report(result: BacktestResult) -> dict[str, Any]:
    """Print a human-readable report and return the metrics dict."""
    stats = summarize(result)
    print("\n===== BACKTEST REPORT =====")
    for key, value in stats.items():
        print(f"{key:>18}: {value}")
    print("===========================\n")
    return stats
