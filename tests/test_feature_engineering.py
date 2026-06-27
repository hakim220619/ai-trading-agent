from __future__ import annotations

import unittest

import pandas as pd

from app.ml.feature_engineering import build_target


class TargetTests(unittest.TestCase):
    def test_target_separates_up_down_and_neutral(self) -> None:
        df = pd.DataFrame(
            {
                "close": [100.0, 102.0, 102.1, 99.0],
                "high": [101.0, 103.0, 103.0, 100.0],
                "low": [99.0, 101.0, 101.0, 98.0],
                "atr": [2.0, 2.0, 2.0, 2.0],
            }
        )
        target = build_target(df, horizon=1, atr_mult=0.5)
        self.assertEqual(target.iloc[0], 1.0)
        self.assertTrue(pd.isna(target.iloc[1]))
        self.assertEqual(target.iloc[2], 0.0)
        self.assertTrue(pd.isna(target.iloc[3]))


if __name__ == "__main__":
    unittest.main()
