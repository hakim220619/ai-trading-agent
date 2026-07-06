from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.mt5.connection import MT5Connection


class MT5ConnectionIdentityTests(unittest.TestCase):
    def test_connect_rejects_terminal_logged_into_different_account(self) -> None:
        fake_mt5 = MagicMock()
        fake_mt5.initialize.return_value = True
        fake_mt5.terminal_info.return_value = SimpleNamespace(connected=True)
        fake_mt5.account_info.return_value = SimpleNamespace(
            login=111, _asdict=lambda: {"login": 111, "server": "Broker", "balance": 10, "currency": "USD"}
        )
        with (
            patch("app.mt5.connection.MT5_AVAILABLE", True),
            patch("app.mt5.connection.mt5", fake_mt5),
            patch("app.mt5.connection.settings.mt5_path", r"C:\\MT5-2\\terminal64.exe"),
            patch("app.mt5.connection.settings.mt5_login", 222),
            patch("app.mt5.connection.settings.mt5_password", "secret"),
            patch("app.mt5.connection.settings.mt5_server", "Broker"),
        ):
            self.assertFalse(MT5Connection().connect())

        fake_mt5.initialize.assert_called_once_with(
            path=r"C:\\MT5-2\\terminal64.exe", login=222, password="secret", server="Broker"
        )
        fake_mt5.shutdown.assert_called_once()


if __name__ == "__main__":
    unittest.main()
