from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.account_manager import AccountProfile
from app.mt5.connection import MT5Connection


class MT5ConnectionIdentityTests(unittest.TestCase):
    def test_login_uses_terminal_path_from_dashboard(self) -> None:
        fake_mt5 = MagicMock()
        fake_mt5.initialize.return_value = True
        fake_mt5.account_info.return_value = SimpleNamespace(login=222)
        with (
            patch("app.mt5.connection.MT5_AVAILABLE", True),
            patch("app.mt5.connection.mt5", fake_mt5),
        ):
            ok, _ = MT5Connection().login(
                222, "secret", "Broker", r"C:\\Selected\\terminal64.exe"
            )

        self.assertTrue(ok)
        fake_mt5.initialize.assert_called_once_with(
            path=r"C:\\Selected\\terminal64.exe",
            login=222,
            password="secret",
            server="Broker",
        )

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
            patch("app.account_manager.account_manager.profile", return_value=AccountProfile(
                "default", "Akun Utama", 222, "Broker", r"C:\\MT5-2\\terminal64.exe"
            )),
            patch("app.account_manager.account_manager.saved_password", return_value="secret"),
        ):
            self.assertFalse(MT5Connection().connect())

        fake_mt5.initialize.assert_called_once_with(
            path=r"C:\\MT5-2\\terminal64.exe", login=222, password="secret", server="Broker"
        )
        fake_mt5.shutdown.assert_called_once()


if __name__ == "__main__":
    unittest.main()
