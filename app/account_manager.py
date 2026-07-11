"""Isolated MT5 account workers managed as local API subprocesses."""
from __future__ import annotations

import json
import os
import base64
import socket
import subprocess
import sys
import time
import ctypes
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def is_account_worker() -> bool:
    return os.getenv("ACCOUNT_WORKER", "0") == "1"


@dataclass
class AccountWorker:
    account_id: str
    label: str
    login: int
    server: str
    terminal_path: str
    port: int
    process: subprocess.Popen[Any]

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


@dataclass
class AccountProfile:
    account_id: str
    label: str
    login: int
    server: str
    terminal_path: str
    protected_password: str = ""
    symbol: str = "XAUUSD"


class _DataBlob(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_byte))]


def _protect_password(password: str) -> str:
    """Encrypt a password for the current Windows user via DPAPI."""
    if not password or os.name != "nt":
        return ""
    raw = password.encode("utf-8")
    buffer = ctypes.create_string_buffer(raw)
    source = _DataBlob(len(raw), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_byte)))
    output = _DataBlob()
    if not ctypes.windll.crypt32.CryptProtectData(ctypes.byref(source), None, None, None, None, 0, ctypes.byref(output)):
        return ""
    try:
        return base64.b64encode(ctypes.string_at(output.pbData, output.cbData)).decode("ascii")
    finally:
        ctypes.windll.kernel32.LocalFree(output.pbData)


def _unprotect_password(protected: str) -> str:
    if not protected or os.name != "nt":
        return ""
    try:
        raw = base64.b64decode(protected)
        buffer = ctypes.create_string_buffer(raw)
        source = _DataBlob(len(raw), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_byte)))
        output = _DataBlob()
        if not ctypes.windll.crypt32.CryptUnprotectData(ctypes.byref(source), None, None, None, None, 0, ctypes.byref(output)):
            return ""
        try:
            return ctypes.string_at(output.pbData, output.cbData).decode("utf-8")
        finally:
            ctypes.windll.kernel32.LocalFree(output.pbData)
    except (ValueError, OSError):
        return ""


class AccountManager:
    def __init__(self) -> None:
        self._workers: dict[str, AccountWorker] = {}
        self._profiles: dict[str, AccountProfile] = {}
        self._lock = Lock()
        self._config_path = Path(os.getenv("ACCOUNTS_CONFIG_PATH", "settings/accounts.json"))
        self._config_mtime_ns: int | None = None
        self._load_profiles()

    def _load_profiles(self) -> None:
        try:
            payload = json.loads(self._config_path.read_text(encoding="utf-8"))
            profiles: dict[str, AccountProfile] = {}
            for item in payload.get("accounts", []):
                profile = AccountProfile(
                    account_id=str(item["account_id"]), label=str(item.get("label") or item["account_id"]),
                    login=int(item["login"]), server=str(item["server"]), terminal_path=str(item["terminal_path"]),
                    protected_password=str(item.get("protected_password", "")),
                    symbol=str(item.get("symbol", "XAUUSD")).strip().upper() or "XAUUSD",
                )
                profiles[profile.account_id] = profile
            self._profiles = profiles
            self._config_mtime_ns = self._config_path.stat().st_mtime_ns
        except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            self._profiles = {}

    def _reload_if_changed(self) -> None:
        """Reload profiles after accounts.json is edited outside the app."""
        try:
            modified = self._config_path.stat().st_mtime_ns
        except OSError:
            modified = None
        if modified != self._config_mtime_ns:
            self._load_profiles()

    def _save_profiles(self) -> None:
        self._config_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"accounts": [profile.__dict__ for profile in self._profiles.values()]}
        temporary = self._config_path.with_suffix(self._config_path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(self._config_path)

    @staticmethod
    def _free_port() -> int:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

    def add(self, account_id: str, label: str, login: int, password: str, server: str, terminal_path: str, *, persist: bool = True) -> AccountWorker:
        account_id = account_id.strip().lower()
        if not account_id or not all(char.isalnum() or char in "-_" for char in account_id):
            raise ValueError("account_id hanya boleh berisi huruf, angka, - atau _")
        if not terminal_path.strip():
            raise ValueError("path terminal MT5 wajib diisi untuk akun tambahan")
        with self._lock:
            existing = self._workers.get(account_id)
            if existing and existing.process.poll() is None:
                raise ValueError(f"account_id {account_id} sudah aktif")
            port = self._free_port()
            env = os.environ.copy()
            # A new account starts with the primary account's strategy, while
            # its own dashboard config remains isolated and can later diverge.
            from app.runtime_config import get_trading_setup

            env.update({
                "ACCOUNT_WORKER": "1",
                "ACCOUNT_ID": account_id,
                "MT5_LOGIN": str(login),
                "MT5_PASSWORD": password,
                "MT5_SERVER": server,
                "MT5_PATH": terminal_path,
                "API_PORT": str(port),
                "AUTO_START": "false",
                "LOG_DIR": f"logs/accounts/{account_id}",
                "DASHBOARD_CONFIG_PATH": f"logs/accounts/{account_id}/dashboard_config.json",
                "DEFAULT_ACTIVE_STRATEGY": str(get_trading_setup()["active_strategy"]),
            })
            process = subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(port)],
                env=env,
                cwd=os.getcwd(),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            worker = AccountWorker(account_id, label.strip() or account_id, login, server.strip(), terminal_path.strip(), port, process)
            self._workers[account_id] = worker
        deadline = time.time() + 60
        observed_login: int | None = None
        while time.time() < deadline:
            if process.poll() is not None:
                break
            try:
                status, body, _ = self.request(account_id, "GET", "/account", timeout=2)
                payload = json.loads(body)
                if status == 200 and payload.get("connected"):
                    observed_login = int((payload.get("info") or {}).get("login") or 0)
                    if observed_login != int(login):
                        break
                    if persist:
                        previous = self._profiles.get(account_id)
                        protected = _protect_password(password) or (previous.protected_password if previous else "")
                        self._profiles[account_id] = AccountProfile(
                            account_id, worker.label, login, server.strip(), terminal_path.strip(), protected,
                            previous.symbol if previous else "XAUUSD",
                        )
                        self._save_profiles()
                    return worker
            except (URLError, ConnectionError, KeyError, TimeoutError):
                time.sleep(0.25)
        self._stop_worker(account_id)
        if observed_login and observed_login != int(login):
            raise RuntimeError(
                f"terminal MT5 memakai akun {observed_login}, bukan akun {login}. "
                "Pastikan setiap akun memakai instalasi/path terminal64.exe yang berbeda."
            )
        raise RuntimeError("worker akun gagal terhubung ke terminal MT5")

    def _stop_worker(self, account_id: str) -> bool:
        with self._lock:
            worker = self._workers.pop(account_id, None)
        if not worker:
            return False
        worker.process.terminate()
        try:
            worker.process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            worker.process.kill()
        return True

    def remove(self, account_id: str) -> bool:
        stopped = self._stop_worker(account_id)
        removed_profile = self._profiles.pop(account_id, None) is not None
        if removed_profile:
            self._save_profiles()
        return stopped or removed_profile

    def restore(self) -> None:
        """Restart every saved account worker after the main application starts."""
        for profile in list(self._profiles.values()):
            if profile.account_id == "default":
                continue
            if self.get(profile.account_id):
                continue
            try:
                self.add(
                    profile.account_id, profile.label, profile.login,
                    _unprotect_password(profile.protected_password), profile.server,
                    profile.terminal_path, persist=False,
                )
            except (ValueError, RuntimeError):
                # Keep the profile visible as offline so it can be reconnected later.
                continue

    def get(self, account_id: str) -> AccountWorker | None:
        worker = self._workers.get(account_id)
        if worker and worker.process.poll() is None:
            return worker
        return None

    def saved_password(self, account_id: str) -> str:
        """Return a saved credential internally without exposing it through the API."""
        profile = self._profiles.get(account_id.strip().lower())
        return _unprotect_password(profile.protected_password) if profile else ""

    def profile(self, account_id: str) -> AccountProfile | None:
        """Return an internal saved profile, including the primary account."""
        return self._profiles.get(account_id.strip().lower())

    def list(self) -> list[dict[str, Any]]:
        self._reload_if_changed()
        primary = self._profiles.get("default")
        default_row = (
            {"account_id": "default", "label": primary.label, "login": primary.login,
             "server": primary.server, "terminal_path": primary.terminal_path,
             "symbol": primary.symbol, "worker": False, "online": True}
            if primary else
            {"account_id": "default", "label": "Akun Utama", "worker": False, "online": True}
        )
        return [
            default_row,
            *[
                {"account_id": profile.account_id, "label": profile.label, "login": profile.login,
                 "server": profile.server, "terminal_path": profile.terminal_path,
                 "symbol": profile.symbol,
                 "password_saved": bool(profile.protected_password), "worker": True,
                 "online": self.get(profile.account_id) is not None}
                for profile in self._profiles.values() if profile.account_id != "default"
            ],
        ]

    def request(self, account_id: str, method: str, path: str, body: bytes | None = None, headers: dict[str, str] | None = None, timeout: float = 30) -> tuple[int, bytes, str]:
        worker = self.get(account_id)
        if not worker:
            raise KeyError(account_id)
        request = Request(worker.base_url + path, data=body, method=method, headers=headers or {})
        try:
            with urlopen(request, timeout=timeout) as response:
                return response.status, response.read(), response.headers.get("content-type", "application/json")
        except HTTPError as exc:
            return exc.code, exc.read(), exc.headers.get("content-type", "application/json")

    def shutdown(self) -> None:
        for account_id in list(self._workers):
            self._stop_worker(account_id)


account_manager = AccountManager()
