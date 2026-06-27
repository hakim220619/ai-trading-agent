from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from app.backtest.day_trade_backtester import run_day_trade_backtest
from app.strategy.day_trade_strategy import DayTradeSignal, generate_day_trade_signal


class DayTradeSignalTests(unittest.TestCase):
    def _buy_row(self) -> pd.Series:
        return pd.Series(
            {
                "close": 105.0,
                "ema20": 103.0,
                "ema50": 101.0,
                "ema200": 95.0,
                "rsi": 60.0,
                "macd_hist": 1.0,
                "breakout_high": 104.0,
                "breakout_low": 90.0,
                "atr": 2.0,
                "atr_ratio": 0.019,
                "trend_strength": 1.0,
                "volume_ratio": 1.2,
                "candle_body": 1.0,
            }
        )

    def test_buy_requires_all_momentum_breakout_filters(self) -> None:
        signal = generate_day_trade_signal(self._buy_row(), 100, 300)
        self.assertEqual(signal.action, "BUY")

    def test_high_spread_blocks_otherwise_valid_entry(self) -> None:
        signal = generate_day_trade_signal(self._buy_row(), 301, 300)
        self.assertEqual(signal.action, "HOLD")
        self.assertIn("spread too high", signal.reasons[0])


class DayTradeBacktestTests(unittest.TestCase):
    @patch("app.strategy.day_trade_strategy.calculate_lot", return_value=0.01)
    @patch("app.backtest.day_trade_backtester._contract", return_value=(0.01, 0.01, 1.0))
    @patch("app.backtest.day_trade_backtester.generate_day_trade_signal")
    def test_backtester_opens_and_closes_simulated_trade(
        self,
        signal_mock,
        _contract_mock,
        _lot_mock,
    ) -> None:
        count = 300
        times = pd.date_range("2026-01-01", periods=count, freq="5min")
        frame = pd.DataFrame(
            {
                "time": times,
                "open": [100.0] * count,
                "high": [100.5] * count,
                "low": [99.5] * count,
                "close": [100.0] * count,
                "volume": [1000.0] * count,
                "spread": [10.0] * count,
            }
        )
        frame.loc[201, "high"] = 104.0

        def signal_for_row(row, _spread, _maximum, preset=None):
            if row.name == 200:
                return DayTradeSignal(action="BUY", price=100.0, atr=1.0)
            return DayTradeSignal()

        signal_mock.side_effect = signal_for_row
        result = run_day_trade_backtest(frame, max_hold_bars=12)
        self.assertEqual(len(result.trades), 1)
        self.assertEqual(result.trades[0].result, "win")
        self.assertGreater(result.end_balance, result.start_balance)


if __name__ == "__main__":
    unittest.main()
