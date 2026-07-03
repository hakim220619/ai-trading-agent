from __future__ import annotations

import unittest
from unittest.mock import patch

from app.strategy.risk_manager import apply_money_limits, build_trade_plan
from app.backtest.backtester import Trade, _trade_pnl


_META = {
    "point": 0.001,
    "tick_value": 1.0,
    "tick_size": 0.001,
    "volume_min": 0.01,
    "volume_max": 100.0,
    "volume_step": 0.01,
    "digits": 3,
    "stops_level": 100,
}


class RiskManagerTests(unittest.TestCase):
    @patch("app.strategy.risk_manager._symbol_meta", return_value=_META)
    def test_buy_plan_has_broker_safe_sl_and_two_to_one_tp(self, _meta) -> None:
        plan = build_trade_plan("BUY", 100.0, 1.0, 1000.0)
        self.assertLess(plan.stop_loss, plan.entry)
        self.assertGreater(plan.take_profit, plan.entry)
        self.assertAlmostEqual(
            plan.take_profit - plan.entry,
            (plan.entry - plan.stop_loss) * 2.0,
            places=3,
        )

    @patch("app.strategy.risk_manager._symbol_meta", return_value=_META)
    def test_sell_plan_has_broker_safe_sl_and_two_to_one_tp(self, _meta) -> None:
        plan = build_trade_plan("SELL", 100.0, 1.0, 1000.0)
        self.assertGreater(plan.stop_loss, plan.entry)
        self.assertLess(plan.take_profit, plan.entry)
        self.assertAlmostEqual(
            plan.entry - plan.take_profit,
            (plan.stop_loss - plan.entry) * 2.0,
            places=3,
        )

    @patch("app.strategy.risk_manager._symbol_meta", return_value=_META)
    def test_fixed_dashboard_distances_override_atr_for_buy(self, _meta) -> None:
        plan = build_trade_plan(
            "BUY", 100.0, 9.0, 1000.0,
            fixed_stop_distance=2.5,
            fixed_take_profit_distance=4.0,
        )
        self.assertEqual(plan.stop_loss, 97.5)
        self.assertEqual(plan.take_profit, 104.0)

    @patch("app.strategy.risk_manager._symbol_meta", return_value=_META)
    def test_fixed_dashboard_distances_override_atr_for_sell(self, _meta) -> None:
        plan = build_trade_plan(
            "SELL", 100.0, 9.0, 1000.0,
            fixed_stop_distance=2.5,
            fixed_take_profit_distance=4.0,
        )
        self.assertEqual(plan.stop_loss, 102.5)
        self.assertEqual(plan.take_profit, 96.0)

    @patch("app.strategy.risk_manager._symbol_meta", return_value={**_META, "stops_level": 0})
    def test_money_limits_are_converted_to_buy_prices(self, _meta) -> None:
        plan = build_trade_plan("BUY", 100.0, 1.0, 1000.0)
        plan.lot = 0.1
        apply_money_limits(plan, "XAUUSD", 1.0, 2.0)
        self.assertEqual(plan.stop_loss, 99.99)
        self.assertEqual(plan.take_profit, 100.02)

    @patch(
        "app.backtest.backtester.connection.symbol_info",
        return_value={"trade_tick_size": 0.01, "trade_tick_value": 0.01},
    )
    def test_backtest_pnl_uses_broker_contract(self, _symbol_info) -> None:
        trade = Trade(
            direction="BUY",
            entry_time=0,
            entry=100_000.0,
            sl=99_000.0,
            tp=102_000.0,
            lot=0.01,
            exit=101_000.0,
        )
        self.assertAlmostEqual(_trade_pnl(trade), 10.0)

    def test_backtest_pnl_subtracts_round_trip_commission(self) -> None:
        trade = Trade(
            direction="BUY",
            entry_time=0,
            entry=100.0,
            sl=99.0,
            tp=102.0,
            lot=1.0,
            exit=101.0,
            pnl_per_price_unit=10.0,
            commission=2.0,
        )
        self.assertAlmostEqual(_trade_pnl(trade), 8.0)


if __name__ == "__main__":
    unittest.main()
