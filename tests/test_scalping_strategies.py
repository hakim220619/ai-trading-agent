from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from app.strategy.scalping_common import market_family
from app.strategy.scalping_m1_strategy import build_m1_plan, generate_m1_signal
from app.strategy.scalping_m5_strategy import generate_m5_signal


class ScalpingStrategyTests(unittest.TestCase):
    def test_market_variants_are_classified(self) -> None:
        self.assertEqual(market_family("BTCUSDm"), "crypto")
        self.assertEqual(market_family("xauusd.pro"), "metals")
        self.assertEqual(market_family("EURUSD"), "standard")

    def test_m1_buy_pullback(self) -> None:
        row = pd.Series({
            "close": 101.0, "low": 100.8, "high": 101.05,
            "ema5": 100.9, "ema13": 100.5, "ema50": 99.0,
            "rsi": 60.0, "macd_hist": 0.2, "atr": 1.0,
            "atr_ratio": 0.005, "volume_ratio": 1.2, "candle_body": 0.4,
            "previous_close": 100.8, "previous_ema5": 100.2,
            "previous_ema13": 100.3, "breakout_high": 100.8,
            "breakout_low": 98.0, "prior_breakout_high": 100.7,
            "prior_breakout_low": 98.0, "micro_trend_strength": 0.4,
            "bb_lower": 99.0, "bb_upper": 102.0,
            "m5_ema20": 100.0, "m5_ema50": 99.0, "m5_ema200": 98.0,
            "m5_rsi": 58.0, "m5_macd_hist": 0.1,
            "h1_ema20": 100.0, "h1_ema50": 99.0, "h1_ema200": 98.0,
            "h1_rsi": 58.0, "h1_macd_hist": 0.1,
        })
        self.assertEqual(generate_m1_signal(row, "XAUUSD", 20, 100).action, "BUY")

    def test_m5_crypto_sell_breakout(self) -> None:
        row = pd.Series({
            "close": 99000.0, "ema9": 99500.0, "ema21": 100000.0,
            "ema50": 101000.0, "ema200": 102000.0, "rsi": 40.0, "macd_hist": -20.0,
            "breakout_high": 103000.0, "breakout_low": 99100.0,
            "atr": 1000.0, "atr_ratio": 0.01, "volume_ratio": 1.2,
            "candle_body": 500.0, "previous_close": 99500.0,
            "previous_ema9": 100100.0, "previous_ema21": 100000.0,
            "ema21_slope": -100.0, "trend_strength": 0.5,
            "close_location": 0.2, "low": 98800.0, "high": 99500.0,
            "bb_lower": 98000.0, "bb_upper": 102000.0,
        })
        self.assertEqual(generate_m5_signal(row, "BTCUSD", 50, 100).action, "SELL")

    def test_spread_guard(self) -> None:
        row = pd.Series({
            "close": 1.2, "ema9": 1.19, "ema21": 1.18, "ema50": 1.17, "ema200": 1.16,
            "rsi": 60.0, "macd_hist": 0.01, "breakout_high": 1.195,
            "breakout_low": 1.15, "atr": 0.005, "atr_ratio": 0.004,
            "volume_ratio": 1.2, "candle_body": 0.003, "previous_close": 1.19,
            "previous_ema9": 1.17, "previous_ema21": 1.18,
            "ema21_slope": 0.001, "trend_strength": 0.3, "close_location": 0.8,
            "low": 1.19, "high": 1.201, "bb_lower": 1.17, "bb_upper": 1.21,
        })
        signal = generate_m5_signal(row, "EURUSD", 101, 100)
        self.assertEqual(signal.action, "HOLD")
        self.assertIn("spread too high", signal.reasons[0])

    @patch("app.strategy.scalping_common.calculate_lot", return_value=0.01)
    def test_m1_plan_uses_market_specific_atr_stop(self, _lot) -> None:
        plan = build_m1_plan("BUY", 100.0, 2.0, 1000.0, "XAUUSD")
        self.assertAlmostEqual(plan.stop_loss, 97.7)
        self.assertAlmostEqual(plan.take_profit, 102.99)


if __name__ == "__main__":
    unittest.main()
