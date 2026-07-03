from __future__ import annotations

import unittest
from unittest.mock import PropertyMock, patch

from app.api.routes import confidence_auto_markets, connection, markets, signal
from app.api.schemas import AutoMarketsRequest


class MarketRouteTests(unittest.TestCase):
    def test_markets_route_returns_open_market_rows(self) -> None:
        rows = [{
            "symbol": "XAUUSD",
            "description": "Gold vs US Dollar",
            "is_open": True,
            "bid": 2320.1,
            "ask": 2320.3,
            "tick_age_minutes": 0.4,
        }]

        with patch.object(type(connection), "connected", new_callable=PropertyMock, return_value=True), patch("app.api.routes.connection.list_markets", return_value=rows) as list_markets:
            result = markets(only_open=True, search="xau", limit=5000, max_tick_age_minutes=180)

        list_markets.assert_called_once_with(
            only_open=True,
            search="xau",
            limit=2000,
            max_tick_age_minutes=180,
        )
        self.assertTrue(result["connected"])
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["markets"][0]["symbol"], "XAUUSD")

    def test_confidence_auto_accepts_dynamic_broker_symbols(self) -> None:
        class FakeBot:
            confidence_auto = True

            def set_auto_symbols(self, symbols: list[str]) -> list[str]:
                return symbols

        with patch("app.api.routes.connection.symbol_info", return_value={"name": "EURUSD"}), patch("app.api.routes._get_bot", return_value=FakeBot()):
            result = confidence_auto_markets(AutoMarketsRequest(symbols=["EURUSD", "XAUUSD.pro"]))

        self.assertTrue(result.ok)
        self.assertEqual(result.detail["symbols"], ["EURUSD", "XAUUSD.pro"])

    def test_signal_route_accepts_symbol_for_market_scan(self) -> None:
        class FakeSignal:
            action = "HOLD"
            reasons: list[str] = []

            def to_dict(self) -> dict:
                return {"action": "HOLD", "confidence": 0.72, "ml": {"buy": 0.72, "sell": 0.28}, "reasons": []}

        class FakeBot:
            PRIMARY_TF = "M5"

            def compute_signal_now(self, timeframe: str, symbol: str) -> FakeSignal:
                self.called = (timeframe, symbol)
                return FakeSignal()

            def preview_trade_plan(self, *_args, **_kwargs):
                return None

        bot = FakeBot()
        with patch("app.api.routes.connection.symbol_info", return_value={"name": "EURUSD"}), patch("app.api.routes._get_bot", return_value=bot), patch("app.api.routes.position_manager.get_open_positions", return_value=[]):
            result = signal(timeframe="M5", symbol="EURUSD")

        self.assertEqual(result.signal["confidence"], 0.72)
        self.assertEqual(bot.called, ("M5", "EURUSD"))


if __name__ == "__main__":
    unittest.main()
