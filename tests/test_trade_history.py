from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.api.routes import trade_history
from app.mt5.confidence_metadata import confidence_comment, parse_confidence_pct, parse_cycle_key
from app.mt5.trade_history import get_capital_curve, summarize_by_open_hour, summarize_closed_deals


class TradeHourAnalysisTests(unittest.TestCase):
    def test_capital_growth_excludes_account_deposits(self) -> None:
        deals = [
            SimpleNamespace(time=1_700_000_000, profit=100.0, commission=0.0, swap=0.0, fee=0.0, type=2),
            SimpleNamespace(time=1_700_000_100, profit=10.0, commission=0.0, swap=0.0, fee=0.0, type=1),
            SimpleNamespace(time=1_700_000_200, profit=20.0, commission=0.0, swap=0.0, fee=0.0, type=2),
            SimpleNamespace(time=1_700_000_300, profit=-18.0, commission=0.0, swap=0.0, fee=0.0, type=1),
        ]
        fake_mt5 = SimpleNamespace(DEAL_TYPE_BALANCE=2, history_deals_get=lambda _start, _end: deals)
        account = {"login": 12345, "server": "Demo", "balance": 112.0, "equity": 115.0, "currency": "USD"}
        with (
            patch("app.mt5.trade_history.MT5_AVAILABLE", True),
            patch("app.mt5.trade_history.connection.ensure_connected", return_value=True),
            patch("app.mt5.trade_history.connection.account_info", return_value=account),
            patch("app.mt5.trade_history.mt5", fake_mt5),
        ):
            result = get_capital_curve()
        self.assertEqual(result["account_login"], 12345)
        self.assertEqual(result["net_deposits"], 120.0)
        self.assertEqual(result["trading_growth"], -8.0)
        self.assertAlmostEqual(result["growth_pct"], -6.67, places=2)

    def test_confidence_comment_round_trip(self) -> None:
        comment = confidence_comment(0.7346, "BUY")
        self.assertEqual(comment, "ai-cf-07346-buy")
        self.assertEqual(parse_confidence_pct(comment), 73.46)

    def test_compact_recovery_comment_survives_broker_limit(self) -> None:
        comment = "rA1b2s3bc650"
        self.assertLessEqual(len(comment), 16)
        self.assertEqual(parse_cycle_key(comment), "A1B2")
        self.assertEqual(parse_confidence_pct(comment), 65.0)

    def test_average_confidence_only_uses_profitable_trades(self) -> None:
        deals = [
            {"net_profit": 5.0, "confidence_pct": 70.0, "commission": 0.0, "swap": 0.0},
            {"net_profit": 3.0, "confidence_pct": 80.0, "commission": 0.0, "swap": 0.0},
            {"net_profit": -2.0, "confidence_pct": 95.0, "commission": 0.0, "swap": 0.0},
            {"net_profit": 1.0, "confidence_pct": None, "commission": 0.0, "swap": 0.0},
        ]
        summary = summarize_closed_deals(deals)
        self.assertEqual(summary["average_profit_confidence_pct"], 75.0)
        self.assertEqual(summary["profit_confidence_trades"], 2)

    def test_groups_profit_and_loss_by_local_open_hour(self) -> None:
        deals = [
            {"open_time": "2026-06-29T10:15:00+00:00", "net_profit": 5.0},
            {"open_time": "2026-06-29T10:45:00+00:00", "net_profit": -2.0},
            {"open_time": "2026-06-29T12:00:00+00:00", "net_profit": 1.0},
        ]
        result = summarize_by_open_hour(deals, timezone_offset_minutes=420)
        hour_three = next(row for row in result["hours"] if row["hour"] == 3)
        self.assertEqual(hour_three["trades"], 2)
        self.assertEqual(hour_three["wins"], 1)
        self.assertEqual(hour_three["losses"], 1)
        self.assertEqual(hour_three["net_profit"], 3.0)
        self.assertEqual(result["best_hour"]["hour"], 3)

    def test_trade_history_route_filters_result(self) -> None:
        deals = [
            {"ticket": 1, "time": "2026-06-29T10:00:00+00:00", "result": "WIN", "net_profit": 5.0, "commission": 0.0, "swap": 0.0},
            {"ticket": 2, "time": "2026-06-29T11:00:00+00:00", "result": "LOSS", "net_profit": -2.0, "commission": 0.0, "swap": 0.0},
            {"ticket": 3, "time": "2026-06-29T12:00:00+00:00", "result": "BE", "net_profit": 0.0, "commission": 0.0, "swap": 0.0},
        ]

        with patch("app.mt5.trade_history.get_closed_deals", return_value=deals):
            result = trade_history(days=31, limit=100, all_account=True, result="loss")

        self.assertEqual(result.days, 31)
        self.assertEqual(result.summary["count"], 1)
        self.assertEqual(result.summary["losses"], 1)
        self.assertEqual(result.deals[0]["ticket"], 2)


if __name__ == "__main__":
    unittest.main()
