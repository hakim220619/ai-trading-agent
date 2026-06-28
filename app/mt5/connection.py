"""MetaTrader 5 connection management.

The ``MetaTrader5`` package only runs on Windows. On other platforms (e.g.
macOS used for development / backtesting) the import is guarded so the rest of
the project still loads. Run live trading on a Windows VPS / Windows VM.
"""
from __future__ import annotations

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

        init_kwargs: dict[str, Any] = {}
        if settings.mt5_path:
            init_kwargs["path"] = settings.mt5_path

        if not mt5.initialize(**init_kwargs):
            logger.error("mt5.initialize() failed: {}", mt5.last_error())
            return False

        # Log in only if credentials were supplied; otherwise rely on the
        # already-logged-in terminal session.
        if settings.mt5_login and settings.mt5_password and settings.mt5_server:
            authorized = mt5.login(
                login=int(settings.mt5_login),
                password=settings.mt5_password,
                server=settings.mt5_server,
            )
            if not authorized:
                logger.error("mt5.login() failed: {}", mt5.last_error())
                mt5.shutdown()
                return False

        self._connected = True
        info = self.account_info()
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
        if self._connected and mt5.terminal_info() is not None:
            return True
        logger.warning("MT5 session lost - attempting reconnect...")
        self._connected = False
        return self.connect()

    def disconnect(self) -> None:
        """Shut down the MT5 terminal session."""
        if MT5_AVAILABLE and self._connected:
            mt5.shutdown()
        self._connected = False
        logger.info("MT5 disconnected.")

    def login(self, login: int, password: str, server: str) -> tuple[bool, str]:
        """Explicitly login from the dashboard without persisting credentials."""
        if not MT5_AVAILABLE:
            return False, "MetaTrader5 package unavailable"
        if self._connected:
            mt5.shutdown()
            self._connected = False
        init_kwargs: dict[str, Any] = {}
        if settings.mt5_path:
            init_kwargs["path"] = settings.mt5_path
        if not mt5.initialize(**init_kwargs):
            return False, f"mt5.initialize failed: {mt5.last_error()}"
        if not mt5.login(login=int(login), password=password, server=server):
            error = mt5.last_error()
            mt5.shutdown()
            return False, f"mt5.login failed: {error}"
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
