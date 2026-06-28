from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from app.mt5.market_data import export_candles_csv
from app.strategy.signal_generator import Signal, confirm_multi_timeframe, generate_signal


def _feature_frame(trend: str = "BUY") -> pd.DataFrame:
    buy = trend == "BUY"
    return pd.DataFrame(
        [
            {
                "time": pd.Timestamp("2026-01-01"),
                "open": 100.0,
                "high": 102.0,
                "low": 99.0,
                "close": 101.0 if buy else 99.0,
                "volume": 1000.0,
                "spread": 12.0,
                "return_1": 0.01,
                "return_3": 0.03,
                "return_5": 0.05,
                "ema20": 100.0 if buy else 100.0,
                "ema50": 99.0 if buy else 101.0,
                "ema200": 98.0 if buy else 102.0,
                "rsi": 60.0 if buy else 40.0,
                "macd": 1.0,
                "macd_signal": 0.5,
                "macd_hist": 0.5,
                "atr": 1.0,
                "bb_middle": 100.0,
                "bb_upper": 103.0,
                "bb_lower": 97.0,
                "candle_body": 1.0,
                "upper_wick": 1.0,
                "lower_wick": 1.0,
                "volume_avg": 900.0,
            }
        ]
    )


class SignalTests(unittest.TestCase):
    @patch("app.strategy.signal_generator._spread_ok", return_value=(True, 12.0))
    @patch(
        "app.strategy.signal_generator.predict_signal",
        return_value={"buy": 0.5, "sell": 0.5, "model": True},
    )
    def test_live_signal_passes_complete_model_features(self, predict, _spread) -> None:
        generate_signal(_feature_frame())
        features = predict.call_args.args[0]
        self.assertEqual(features["return_1"], 0.01)
        self.assertEqual(features["return_3"], 0.03)
        self.assertEqual(features["return_5"], 0.05)
        self.assertEqual(features["spread"], 12.0)

    def test_multi_timeframe_can_block_opposite_context(self) -> None:
        signal = Signal(action="BUY", reasons=[])
        result = confirm_multi_timeframe(
            signal,
            {"M5": _feature_frame("BUY"), "M15": _feature_frame("SELL")},
            primary="M5",
        )
        self.assertEqual(result.action, "HOLD")
        self.assertIn("MTF blocked BUY", result.reasons[-1])

    @patch("app.strategy.signal_generator._spread_ok", return_value=(False, 700.0))
    @patch(
        "app.strategy.signal_generator.predict_signal",
        return_value={"buy": 0.836, "sell": 0.165, "model": True},
    )
    def test_hold_from_high_spread_keeps_ml_confidence(self, _predict, _spread) -> None:
        signal = generate_signal(_feature_frame())
        self.assertEqual(signal.action, "HOLD")
        self.assertEqual(signal.confidence, 0.836)
        self.assertIn("spread too high", signal.reasons[0])

    @patch("app.strategy.signal_generator.sr.is_near_resistance", return_value=False)
    @patch("app.strategy.signal_generator.sr.is_near_support", return_value=True)
    @patch("app.strategy.signal_generator._spread_ok", return_value=(True, 12.0))
    @patch(
        "app.strategy.signal_generator.predict_signal",
        return_value={"buy": 0.5, "sell": 0.5, "model": False},
    )
    def test_technical_signal_works_before_model_is_trained(
        self,
        _predict,
        _spread,
        _support,
        _resistance,
    ) -> None:
        signal = generate_signal(_feature_frame("BUY"))
        self.assertEqual(signal.action, "BUY")
        self.assertIn("near support", signal.reasons[0])


class ExportTests(unittest.TestCase):
    @patch("app.mt5.market_data.get_candles", return_value=_feature_frame())
    def test_export_writes_training_csv(self, _get_candles) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "candles.csv"
            result = export_candles_csv(path=str(output), count=200)
            self.assertTrue(output.exists())
            self.assertEqual(result["rows"], 1)
            self.assertIn("close", pd.read_csv(output).columns)


if __name__ == "__main__":
    unittest.main()
