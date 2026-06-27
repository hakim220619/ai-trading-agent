from __future__ import annotations

import unittest

import pandas as pd

from app.strategy.support_resistance import detect_levels


class MarketStructureTests(unittest.TestCase):
    def test_bullish_bos_after_higher_high_and_higher_low(self) -> None:
        frame = pd.DataFrame(
            {
                "high": [10, 11, 10, 12, 11, 12, 14],
                "low": [8, 9, 8.5, 10, 9.5, 10.5, 12],
                "close": [9, 10.5, 9, 11.5, 10, 11.5, 13],
            }
        )
        levels = detect_levels(frame, left=1, right=1)
        self.assertEqual(levels.structure_trend, "bullish")
        self.assertEqual(levels.bos, "bullish")
        self.assertEqual(levels.bos_level, 12.0)
        self.assertIsNone(levels.choch)

    def test_bullish_choch_when_bearish_structure_breaks_high(self) -> None:
        frame = pd.DataFrame(
            {
                "high": [10, 12, 10, 11, 9, 10.5, 12],
                "low": [8, 9, 8, 8.5, 7, 8, 9],
                "close": [9, 11, 9, 10, 8, 10.5, 11.5],
            }
        )
        levels = detect_levels(frame, left=1, right=1)
        self.assertEqual(levels.structure_trend, "bearish")
        self.assertEqual(levels.choch, "bullish")
        self.assertEqual(levels.choch_level, 11.0)
        self.assertIsNone(levels.bos)


if __name__ == "__main__":
    unittest.main()
