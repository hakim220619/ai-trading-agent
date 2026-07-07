import unittest
from unittest.mock import patch

from app.strategy.recovery_m1_strategy import RecoveryM1Strategy


class _Result:
    def __init__(self, ok=True, message="ok"):
        self.ok = ok
        self.message = message

    def to_dict(self):
        return {"ok": self.ok, "message": self.message}


class RecoveryM1StrategyTests(unittest.TestCase):
    def setUp(self):
        self.scalping_setup = patch(
            "app.strategy.recovery_m1_strategy.get_scalping_setup",
            return_value={
                "confidence_threshold": 0.50,
                "base_lot": 0.01,
                "second_lot": 0.03,
                "lot_multiplier": 2.0,
                "max_lot": 0.48,
                "initial_loss_money": 3.0,
                "loss_increment_money": 2.0,
                "basket_profit_target": 0.50,
                "basket_loss_limit": 0.0,
                "basket_loss_limit_enabled": False,
                "daily_profit_target": 0.0,
                "daily_profit_target_enabled": False,
                "daily_loss_limit": 0.0,
                "daily_loss_limit_enabled": False,
            },
        )
        self.scalping_setup_mock = self.scalping_setup.start()
        self.addCleanup(self.scalping_setup.stop)
        self.uuid = patch("app.strategy.recovery_m1_strategy.uuid.uuid4")
        uuid_mock = self.uuid.start()
        uuid_mock.return_value.hex = "abc123999999"
        self.addCleanup(self.uuid.stop)

    def test_starts_buy_then_reverses_with_incremented_loss_limit(self):
        strategy = RecoveryM1Strategy()
        with patch("app.strategy.recovery_m1_strategy.get_open_positions", return_value=[]), \
             patch("app.strategy.recovery_m1_strategy.order_executor.configured_minimum_lot", return_value=0.01), \
             patch("app.strategy.recovery_m1_strategy.order_executor.open_buy", return_value=_Result()) as buy:
            first = strategy.tick("XAUUSD", initial_direction="BUY")
        self.assertEqual(first["action"], "OPEN_BUY")
        self.assertEqual(first["loss_limit_money"], 3.0)
        buy.assert_called_once_with(
            "XAUUSD", 0.01, 0.0, 0.0, comment="rabc1s1bc000", enforce_spread=False,
            allow_duplicate=True, enforce_position_limit=False,
        )

    def test_first_position_waits_until_confidence_has_a_winner(self):
        strategy = RecoveryM1Strategy()
        with patch("app.strategy.recovery_m1_strategy.get_open_positions", return_value=[]), \
             patch("app.strategy.recovery_m1_strategy.order_executor.open_buy") as buy, \
             patch("app.strategy.recovery_m1_strategy.order_executor.open_sell") as sell:
            result = strategy.tick("XAUUSD", initial_direction=None)
        self.assertEqual(result["action"], "WAIT_CONFIDENCE")
        buy.assert_not_called()
        sell.assert_not_called()

        losing = [{"ticket": 10, "comment": "m1rec-1-buy", "type_str": "BUY", "profit": -3.0}]
        with patch("app.strategy.recovery_m1_strategy.get_open_positions", return_value=losing), \
             patch("app.strategy.recovery_m1_strategy.order_executor.close_position_ticket") as close, \
             patch("app.strategy.recovery_m1_strategy.order_executor.configured_minimum_lot", return_value=0.01), \
             patch("app.strategy.recovery_m1_strategy.order_executor.open_sell", return_value=_Result()) as sell:
            second = strategy.tick("XAUUSD")
        self.assertEqual(second["action"], "OPEN_SELL")
        self.assertEqual(second["loss_limit_money"], 5.0)
        close.assert_not_called()
        sell.assert_called_once_with(
            "XAUUSD", 0.03, 0.0, 0.0, comment="rabc1s2sc000", enforce_spread=False,
            allow_duplicate=True, enforce_position_limit=False,
        )

        positions = [
            losing[0],
            {"ticket": 11, "comment": "m1rec-2-sell", "type_str": "SELL", "profit": -5.0},
        ]
        with patch("app.strategy.recovery_m1_strategy.get_open_positions", return_value=positions), \
             patch("app.strategy.recovery_m1_strategy.order_executor.configured_minimum_lot", return_value=0.01), \
             patch("app.strategy.recovery_m1_strategy.order_executor.open_buy", return_value=_Result()) as buy:
            third = strategy.tick("XAUUSD")
        self.assertEqual(third["action"], "OPEN_BUY")
        buy.assert_called_once_with(
            "XAUUSD", 0.06, 0.0, 0.0, comment="rabc1s3bc000", enforce_spread=False,
            allow_duplicate=True, enforce_position_limit=False,
        )

    def test_empty_basket_always_resets_next_order_to_base_lot(self):
        strategy = RecoveryM1Strategy()
        strategy._steps["XAUUSD"] = 5
        strategy._directions["XAUUSD"] = "SELL"
        with patch("app.strategy.recovery_m1_strategy.get_open_positions", return_value=[]), \
             patch("app.strategy.recovery_m1_strategy.order_executor.open_buy", return_value=_Result()) as buy:
            result = strategy.tick("XAUUSD", initial_direction="BUY")
        self.assertEqual(result["step"], 1)
        self.assertEqual(result["lot"], 0.01)
        buy.assert_called_once_with(
            "XAUUSD", 0.01, 0.0, 0.0, comment="rabc1s1bc000", enforce_spread=False,
            allow_duplicate=True, enforce_position_limit=False,
        )

    def test_btc_minimum_lot_is_effective_base_for_multiplier(self):
        setup = {
            "base_lot": 0.01,
            "second_lot": 0.03,
            "lot_multiplier": 2.0,
        }
        self.assertEqual(RecoveryM1Strategy._lot_for_step(1, setup, 0.05), 0.05)
        self.assertEqual(RecoveryM1Strategy._lot_for_step(2, setup, 0.05), 0.10)
        self.assertEqual(RecoveryM1Strategy._lot_for_step(3, setup, 0.05), 0.20)

    def test_explicit_larger_second_lot_is_preserved(self):
        setup = {
            "base_lot": 0.01,
            "second_lot": 0.03,
            "lot_multiplier": 2.0,
        }
        self.assertEqual(RecoveryM1Strategy._lot_for_step(2, setup, 0.01), 0.03)
        self.assertEqual(RecoveryM1Strategy._lot_for_step(3, setup, 0.01), 0.06)

    def test_network_failed_counter_retries_even_after_loss_recovers(self):
        strategy = RecoveryM1Strategy()
        losing = [{"ticket": 10, "comment": "m1rec-1-buy", "type_str": "BUY", "profit": -3.0}]
        recovered = [{"ticket": 10, "comment": "m1rec-1-buy", "type_str": "BUY", "profit": -1.0}]
        with patch("app.strategy.recovery_m1_strategy.get_open_positions", side_effect=[losing, recovered]), \
             patch("app.strategy.recovery_m1_strategy.order_executor.configured_minimum_lot", return_value=0.01), \
             patch("app.strategy.recovery_m1_strategy.order_executor.open_sell", side_effect=[
                 _Result(False, "retcode=10031 absence of network connection"), _Result(True, "ok")
             ]) as sell:
            failed = strategy.tick("XAUUSD")
            retried = strategy.tick("XAUUSD")
        self.assertFalse(failed["ok"])
        self.assertTrue(retried["ok"])
        self.assertEqual(retried["step"], 2)
        self.assertEqual(sell.call_count, 2)

    def test_profit_below_half_dollar_stays_open(self):
        strategy = RecoveryM1Strategy()
        winning = [{"ticket": 11, "comment": "m1rec-3-sell", "type_str": "SELL", "profit": 0.01}]
        with patch("app.strategy.recovery_m1_strategy.get_open_positions", return_value=winning), \
             patch("app.strategy.recovery_m1_strategy.order_executor.close_position_ticket") as close:
            result = strategy.tick("XAUUSD")
        self.assertEqual(result["action"], "HOLD")
        close.assert_not_called()

    def test_total_profit_above_half_dollar_closes_immediately(self):
        strategy = RecoveryM1Strategy()
        active = lambda profit: [{"ticket": 12, "comment": "m1rec-1-sell", "type_str": "SELL", "profit": profit}]
        with patch("app.strategy.recovery_m1_strategy.get_open_positions", return_value=active(0.80)), \
             patch("app.strategy.recovery_m1_strategy.order_executor.close_position_ticket", return_value=_Result()) as close:
            closed = strategy.tick("XAUUSD")
        self.assertEqual(closed["action"], "CLOSE_BASKET_PROFIT")
        close.assert_called_once_with(12, "XAUUSD")

    def test_counter_basket_closes_when_total_profit_exceeds_half_dollar(self):
        strategy = RecoveryM1Strategy()
        positions = [
            {"ticket": 20, "comment": "m1rec-1-sell", "type_str": "SELL", "profit": -3.0},
            {"ticket": 21, "comment": "m1rec-2-buy", "type_str": "BUY", "profit": 3.80},
        ]
        with patch("app.strategy.recovery_m1_strategy.get_open_positions", return_value=positions), \
             patch("app.strategy.recovery_m1_strategy.order_executor.close_position_ticket", return_value=_Result()) as close:
            result = strategy.tick("XAUUSD")
        self.assertEqual(result["action"], "CLOSE_BASKET_PROFIT")
        self.assertAlmostEqual(result["basket_profit"], 0.80)
        self.assertEqual(close.call_count, 2)

    def test_counter_basket_closes_when_session_loss_limit_is_reached(self):
        setup = self.scalping_setup_mock.return_value.copy()
        setup.update({"basket_loss_limit": 6.0, "basket_loss_limit_enabled": True})
        self.scalping_setup_mock.return_value = setup
        strategy = RecoveryM1Strategy()
        positions = [
            {"ticket": 30, "comment": "m1rec-1-sell", "type_str": "SELL", "profit": -3.5},
            {"ticket": 31, "comment": "m1rec-2-buy", "type_str": "BUY", "profit": -2.6},
        ]
        with patch("app.strategy.recovery_m1_strategy.get_open_positions", return_value=positions), \
             patch("app.strategy.recovery_m1_strategy.order_executor.close_position_ticket", return_value=_Result()) as close:
            result = strategy.tick("XAUUSD")
        self.assertEqual(result["action"], "CLOSE_BASKET_LOSS_LIMIT")
        self.assertAlmostEqual(result["basket_profit"], -6.10)
        self.assertEqual(close.call_count, 2)


if __name__ == "__main__":
    unittest.main()
