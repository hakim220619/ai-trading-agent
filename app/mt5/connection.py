"""MetaTrader 5 connection management.

The ``MetaTrader5`` package only runs on Windows. On other platforms (e.g.
macOS used for development / backtesting) the import is guarded so the rest of
the project still loads. Run live trading on a Windows VPS / Windows VM.
"""
from __future__ import annotations

from datetime import datetime, timezone
from collections import deque
import os
import platform
from typing import Any

from app.config import settings
from app.utils.logger import logger

# --- Guarded import of the Windows-only MetaTrader5 package -----------------
try:
    import MetaTrader5 as mt5  # type: ignore

    MT5_AVAILABLE = True
except Exception as exc:  # pragma: no cover - depends on platform
    mt5 = None  # type: ignore
    MT5_AVAILABLE = False
    logger.warning(
        "MetaTrader5 package not available ({}). "
        "Live/data features are disabled; backtest & training still work. "
        "Platform={}",
        exc,
        platform.system(),
    )


# --- Timeframe mapping (string -> MT5 constant) ----------------------------
def timeframe_map() -> dict[str, int]:
    """Map human timeframe strings to MT5 timeframe constants.

    Returns an empty mapping when MT5 is unavailable.
    """
    if not MT5_AVAILABLE:
        return {}
    return {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
    }


class MT5Connection:
    """Wrapper that owns the MT5 terminal session and supports reconnect."""

    def __init__(self) -> None:
        self._connected: bool = False
        self._user_logged_out: bool = False
        self._ping_samples: deque[float] = deque(maxlen=20)

    @property
    def available(self) -> bool:
        """True only when the MetaTrader5 package is importable on this OS."""
        return MT5_AVAILABLE

    @property
    def connected(self) -> bool:
        return self._connected

    def connect(self) -> bool:
        """Initialise the MT5 terminal and log in.

        Returns True on success. Credentials come from settings (.env).
        """
        if not MT5_AVAILABLE:
            logger.error("Cannot connect: MetaTrader5 package not installed on this OS.")
            return False
        if self._user_logged_out:
            logger.info("MT5 auto-connect paused after dashboard logout.")
            return False

        from app.account_manager import account_manager

        account_id = os.getenv("ACCOUNT_ID", "default").strip().lower() or "default"
        profile = account_manager.profile(account_id)
        password = account_manager.saved_password(account_id)
        init_kwargs: dict[str, Any] = {}
        if profile and profile.terminal_path:
            init_kwargs["path"] = profile.terminal_path
        if profile and password and profile.server:
            init_kwargs.update({
                "login": int(profile.login),
                "password": password,
                "server": profile.server,
            })

        if not mt5.initialize(**init_kwargs):
            logger.error("mt5.initialize() failed: {}", mt5.last_error())
            return False

        terminal = mt5.terminal_info()
        if terminal is None or not bool(getattr(terminal, "connected", False)):
            self._connected = False
            logger.error("MT5 initialized but broker connection is offline: {}", mt5.last_error())
            return False
        self._connected = True
        account = mt5.account_info()
        info = account._asdict() if account is not None else None
        if profile and (not info or int(info.get("login") or 0) != int(profile.login)):
            actual = info.get("login") if info else None
            logger.error("MT5 account mismatch: requested={} active={}", profile.login, actual)
            mt5.shutdown()
            self._connected = False
            return False
        if info:
            logger.success(
                "Connected to MT5. Account={} Server={} Balance={} {}",
                info.get("login"),
                info.get("server"),
                info.get("balance"),
                info.get("currency"),
            )
        return True

    def ensure_connected(self) -> bool:
        """Reconnect automatically if the terminal dropped the session."""
        if not MT5_AVAILABLE:
            return False
        if self._user_logged_out:
            return False
        terminal = mt5.terminal_info() if self._connected else None
        if terminal is not None and bool(getattr(terminal, "connected", False)):
            return True
        logger.warning("MT5 session lost - attempting reconnect...")
        self._connected = False
        return self.connect()

    def connection_health(self, symbol: str | None = None) -> dict[str, Any]:
        """Return broker-link latency and fresh-price health for the dashboard."""
        if not self.ensure_connected():
            return {"broker_connected": False, "stable": False, "ping_ms": None, "tick_age_seconds": None}
        terminal = mt5.terminal_info()
        broker_connected = bool(terminal is not None and getattr(terminal, "connected", False))
        ping_raw = float(getattr(terminal, "ping_last", 0.0) or 0.0) if terminal is not None else 0.0
        ping_ms = ping_raw / 1000.0 if ping_raw > 0 else None  # MT5 exposes microseconds
        if ping_ms is not None:
            self._ping_samples.append(ping_ms)

        selected = symbol or settings.symbol
        tick = mt5.symbol_info_tick(selected) if broker_connected else None
        tick_time_msc = float(getattr(tick, "time_msc", 0.0) or 0.0) if tick is not None else 0.0
        tick_age = max(0.0, datetime.now(timezone.utc).timestamp() - tick_time_msc / 1000.0) if tick_time_msc else None
        samples = list(self._ping_samples)
        jitter_ms = max(samples) - min(samples) if len(samples) >= 2 else 0.0
        stable = bool(
            broker_connected
            and ping_ms is not None and ping_ms <= 500
            and jitter_ms <= 250
            and tick_age is not None and tick_age <= 10
        )
        return {
            "broker_connected": broker_connected,
            "stable": stable,
            "ping_ms": round(ping_ms, 1) if ping_ms is not None else None,
            "jitter_ms": round(jitter_ms, 1),
            "tick_age_seconds": round(tick_age, 1) if tick_age is not None else None,
            "samples": len(samples),
        }

    def disconnect(self) -> None:
        """Shut down the MT5 terminal session."""
        if MT5_AVAILABLE and self._connected:
            mt5.shutdown()
        self._connected = False
        logger.info("MT5 disconnected.")

    def login(
        self,
        login: int,
        password: str,
        server: str,
        terminal_path: str | None = None,
    ) -> tuple[bool, str]:
        """Explicitly login from the dashboard without persisting credentials."""
        if not MT5_AVAILABLE:
            return False, "MetaTrader5 package unavailable"
        if self._connected:
            mt5.shutdown()
            self._connected = False
        init_kwargs: dict[str, Any] = {}
        from app.account_manager import account_manager

        account_id = os.getenv("ACCOUNT_ID", "default").strip().lower() or "default"
        profile = account_manager.profile(account_id)
        selected_path = (terminal_path or (profile.terminal_path if profile else "") or "").strip()
        if selected_path:
            init_kwargs["path"] = selected_path
        init_kwargs.update({"login": int(login), "password": password, "server": server})
        if not mt5.initialize(**init_kwargs):
            return False, f"mt5.initialize failed: {mt5.last_error()}"
        info = mt5.account_info()
        actual_login = int(getattr(info, "login", 0) or 0) if info is not None else 0
        if actual_login != int(login):
            error = f"akun aktif {actual_login or '-'}, seharusnya {login}"
            mt5.shutdown()
            return False, f"verifikasi login MT5 gagal: {error}"
        self._user_logged_out = False
        self._connected = True
        return True, "login MT5 berhasil"

    def logout(self) -> None:
        """Disconnect and suppress automatic reconnection until explicit login."""
        if MT5_AVAILABLE:
            mt5.shutdown()
        self._connected = False
        self._user_logged_out = True
        logger.info("MT5 logged out from dashboard.")

    # --- Info helpers -------------------------------------------------------
    def account_info(self) -> dict[str, Any] | None:
        """Return account info as a dict, or None on failure."""
        if not self.ensure_connected():
            return None
        info = mt5.account_info()
        if info is None:
            logger.error("account_info() returned None: {}", mt5.last_error())
            return None
        return info._asdict()

    def symbol_info(self, symbol: str | None = None) -> dict[str, Any] | None:
        """Return symbol info as a dict, selecting the symbol if needed."""
        if not self.ensure_connected():
            return None
        symbol = symbol or settings.symbol
        if not mt5.symbol_select(symbol, True):
            logger.error("Failed to select symbol {}: {}", symbol, mt5.last_error())
            return None
        info = mt5.symbol_info(symbol)
        if info is None:
            logger.error("symbol_info({}) returned None: {}", symbol, mt5.last_error())
            return None
        return info._asdict()

    def list_markets(
        self,
        only_open: bool = True,
        search: str | None = None,
        limit: int = 500,
        max_tick_age_minutes: int = 180,
    ) -> list[dict[str, Any]]:
        """Return broker symbols with a best-effort open/tradable status."""
        if not self.ensure_connected():
            return []
        symbols = mt5.symbols_get() or []
        query = (search or "").strip().lower()
        now = datetime.now(timezone.utc)
        rows: list[dict[str, Any]] = []
        for symbol in symbols:
            info = symbol._asdict()
            name = str(info.get("name") or "")
            if query:
                haystack = " ".join(str(info.get(key) or "") for key in ("name", "path", "description")).lower()
                if query not in haystack:
                    continue
            tick = mt5.symbol_info_tick(name)
            tick_data = tick._asdict() if tick is not None else {}
            trade_mode = int(info.get("trade_mode") or 0)
            bid = float(tick_data.get("bid") or 0.0)
            ask = float(tick_data.get("ask") or 0.0)
            tick_time = tick_data.get("time")
            tick_age_minutes = None
            if tick_time:
                tick_age_minutes = max(0.0, (now - datetime.fromtimestamp(int(tick_time), tz=timezone.utc)).total_seconds() / 60)
            tick_active = bid > 0 and ask > 0 and (tick_age_minutes is None or tick_age_minutes <= max_tick_age_minutes)
            is_open = trade_mode != 0 and tick_active
            if only_open and not is_open:
                continue
            rows.append({
                "symbol": name,
                "description": info.get("description"),
                "path": info.get("path"),
                "visible": bool(info.get("visible")),
                "trade_mode": trade_mode,
                "trade_mode_label": "disabled" if trade_mode == 0 else "tradable",
                "is_open": is_open,
                "bid": bid or None,
                "ask": ask or None,
                "spread": info.get("spread"),
                "digits": info.get("digits"),
                "tick_time": datetime.fromtimestamp(int(tick_time), tz=timezone.utc).isoformat() if tick_time else None,
                "tick_age_minutes": round(tick_age_minutes, 1) if tick_age_minutes is not None else None,
            })
            if len(rows) >= max(1, limit):
                break
        rows.sort(key=lambda row: (not bool(row["is_open"]), str(row["symbol"])))
        return rows

    def get_spread_points(self, symbol: str | None = None) -> float | None:
        """Current spread in points (None on failure)."""
        symbol = symbol or settings.symbol
        info = self.symbol_info(symbol)
        if info is None:
            return None
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return float(info.get("spread", 0))
        point = info.get("point", 0.0) or 0.0
        if point <= 0:
            return float(info.get("spread", 0))
        return round((tick.ask - tick.bid) / point, 1)

    def trading_status(self) -> dict[str, bool]:
        """Return terminal/account permissions relevant to automated orders."""
        status = {
            "terminal_trade_allowed": False,
            "account_trade_allowed": False,
            "trade_api_disabled": True,
        }
        if not self.ensure_connected():
            return status
        terminal = mt5.terminal_info()
        account = mt5.account_info()
        if terminal is not None:
            status["terminal_trade_allowed"] = bool(terminal.trade_allowed)
            status["trade_api_disabled"] = bool(terminal.tradeapi_disabled)
        if account is not None:
            status["account_trade_allowed"] = bool(account.trade_allowed)
        return status

    def can_trade(self) -> bool:
        """True only when both MT5 and the logged-in account allow trading."""
        status = self.trading_status()
        return (
            status["terminal_trade_allowed"]
            and status["account_trade_allowed"]
            and not status["trade_api_disabled"]
        )


# Shared, lazily-used singleton connection.
connection = MT5Connection()
