from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.account_manager import AccountManager


class AccountManagerPersistenceTests(unittest.TestCase):
    def test_saved_profile_is_loaded_and_removed_explicitly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "accounts.json"
            config.write_text(json.dumps({"accounts": [{
                "account_id": "akun-2",
                "label": "Akun Dua",
                "login": 12345,
                "server": "Broker-Demo",
                "terminal_path": r"C:\\MT5-2\\terminal64.exe",
                "protected_password": "encrypted",
            }]}), encoding="utf-8")
            with patch.dict(os.environ, {"ACCOUNTS_CONFIG_PATH": str(config)}):
                manager = AccountManager()
                account = manager.list()[1]
                self.assertEqual(account["account_id"], "akun-2")
                self.assertFalse(account["online"])

                manager.shutdown()
                self.assertEqual(len(manager.list()), 2)

                self.assertTrue(manager.remove("akun-2"))
                self.assertEqual(manager.list(), [{
                    "account_id": "default", "label": "Akun Utama",
                    "worker": False, "online": True,
                }])
                self.assertEqual(json.loads(config.read_text(encoding="utf-8")), {"accounts": []})


if __name__ == "__main__":
    unittest.main()
