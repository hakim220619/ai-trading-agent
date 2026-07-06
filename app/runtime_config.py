"""Small persistent runtime configuration edited from the dashboard."""
from __future__ import annotations

import json
import threading
from pathlib import Path


_PATH = Path("logs/dashboard_config.json")
_LOCK = threading.Lock()
_DEFAULTS = {
    "BTCUSD": {"stop_loss_money": 0.0, "take_profit_money": 0.0},
    "XAUUSD": {"stop_loss_money": 0.0, "take_profit_money": 0.0},
}
_SETUP_DEFAULTS = {
    "active_strategy": "confidence_m5",
    "confidence_threshold": 0.65,
    "max_open_positions": 3,
    "btcusd_min_lot": 0.05,
    "xauusd_min_lot": 0.01,
    "trailing_stop": True,
    "trailing_profit_step_money": 1.0,
    "daily_profit_limit_enabled": False,
    "daily_profit_limit_money": 0.0,
    "daily_lot_limit_enabled": False,
    "daily_lot_limit": 0.0,
}
_SCALPING_DEFAULTS = {
    "confidence_threshold": 0.50,
    "base_lot": 0.01,
    "second_lot": 0.03,
    "lot_multiplier": 2.0,
    "max_lot": 0.48,
    "initial_loss_money": 3.0,
    "loss_increment_money": 2.0,
    "basket_profit_target": 0.50,
    "daily_profit_target": 0.0,
    "daily_profit_target_enabled": False,
}


def _load() -> dict[str, dict[str, float]]:
    data = {symbol: values.copy() for symbol, values in _DEFAULTS.items()}
    try:
        saved = json.loads(_PATH.read_text(encoding="utf-8"))
        for symbol in data:
            values = saved.get(symbol, {})
            for key in ("stop_loss_money", "take_profit_money"):
                data[symbol][key] = max(0.0, float(values.get(key, 0.0)))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        pass
    return data


_CONFIG = _load()


def _load_setup() -> dict[str, float | int | bool | str]:
    try:
        saved = json.loads(_PATH.read_text(encoding="utf-8")).get("trading_setup", {})
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        saved = {}
    return {**_SETUP_DEFAULTS, **{key: saved[key] for key in _SETUP_DEFAULTS if key in saved}}


_SETUP = _load_setup()


def _normalize_scalping_symbol(symbol: str) -> str:
    upper = symbol.upper()
    if "BTCUSD" in upper:
        return "BTCUSD"
    if "XAUUSD" in upper or "GOLD" in upper:
        return "XAUUSD"
    raise ValueError("konfigurasi scalping hanya tersedia untuk BTCUSD atau XAUUSD")


def _load_scalping_setups() -> dict[str, dict[str, float | bool]]:
    try:
        saved = json.loads(_PATH.read_text(encoding="utf-8")).get("scalping_setup", {})
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        saved = {}
    # Migrate the former shared flat profile by copying it to both instruments.
    legacy = saved if isinstance(saved, dict) and "base_lot" in saved else {}
    result: dict[str, dict[str, float | bool]] = {}
    for symbol in _DEFAULTS:
        source = saved.get(symbol, legacy) if isinstance(saved, dict) else {}
        loaded = _SCALPING_DEFAULTS.copy()
        if isinstance(source, dict):
            for key, default in _SCALPING_DEFAULTS.items():
                if key in source:
                    loaded[key] = bool(source[key]) if isinstance(default, bool) else float(source[key])
        result[symbol] = loaded
    return result


_SCALPING_SETUPS = _load_scalping_setups()


def _payload() -> dict:
    return {**_CONFIG, "trading_setup": _SETUP, "scalping_setup": _SCALPING_SETUPS}


def _persist() -> None:
    _PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = _PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(_payload(), indent=2), encoding="utf-8")
    temporary.replace(_PATH)


def get_symbol_risk(symbol: str) -> dict[str, float]:
    with _LOCK:
        return _CONFIG.get(symbol.upper(), {"stop_loss_money": 0.0, "take_profit_money": 0.0}).copy()


def get_all_symbol_risk() -> dict[str, dict[str, float]]:
    with _LOCK:
        return {symbol: values.copy() for symbol, values in _CONFIG.items()}


def get_trading_setup() -> dict[str, float | int | bool | str]:
    with _LOCK:
        return _SETUP.copy()


def get_scalping_setup(symbol: str = "BTCUSD") -> dict[str, float | bool]:
    with _LOCK:
        return _SCALPING_SETUPS[_normalize_scalping_symbol(symbol)].copy()


def get_all_scalping_setups() -> dict[str, dict[str, float | bool]]:
    with _LOCK:
        return {symbol: values.copy() for symbol, values in _SCALPING_SETUPS.items()}


def save_scalping_setup(symbol: str, values: dict) -> dict[str, float | bool]:
    symbol = _normalize_scalping_symbol(symbol)
    normalized = {
        "confidence_threshold": min(0.99, max(0.50, float(values["confidence_threshold"]))),
        "base_lot": max(0.01, float(values["base_lot"])),
        "second_lot": max(0.01, float(values["second_lot"])),
        "lot_multiplier": max(1.0, float(values["lot_multiplier"])),
        "max_lot": max(0.01, float(values["max_lot"])),
        "initial_loss_money": max(0.01, float(values["initial_loss_money"])),
        "loss_increment_money": max(0.01, float(values["loss_increment_money"])),
        "basket_profit_target": max(0.01, float(values["basket_profit_target"])),
        "daily_profit_target": max(0.0, float(values["daily_profit_target"])),
        "daily_profit_target_enabled": bool(values["daily_profit_target_enabled"]),
    }
    with _LOCK:
        _SCALPING_SETUPS[symbol].update(normalized)
        _persist()
        return _SCALPING_SETUPS[symbol].copy()


def save_trading_setup(values: dict) -> dict[str, float | int | bool | str]:
    active_strategy = str(values.get("active_strategy", _SETUP.get("active_strategy", "confidence_m5")))
    if active_strategy not in {"confidence_m5", "recovery_m1"}:
        raise ValueError("strategi tidak dikenal")
    normalized = {
        "active_strategy": active_strategy,
        "confidence_threshold": min(0.99, max(0.5, float(values["confidence_threshold"]))),
        "max_open_positions": min(20, max(1, int(values["max_open_positions"]))),
        "btcusd_min_lot": max(0.01, float(values["btcusd_min_lot"])),
        "xauusd_min_lot": max(0.01, float(values["xauusd_min_lot"])),
        "trailing_stop": bool(values["trailing_stop"]),
        "trailing_profit_step_money": max(0.01, float(values["trailing_profit_step_money"])),
        "daily_profit_limit_enabled": bool(values["daily_profit_limit_enabled"]),
        "daily_profit_limit_money": max(0.0, float(values["daily_profit_limit_money"])),
        "daily_lot_limit_enabled": bool(values["daily_lot_limit_enabled"]),
        "daily_lot_limit": max(0.0, float(values["daily_lot_limit"])),
    }
    with _LOCK:
        _SETUP.update(normalized)
        _persist()
        return _SETUP.copy()


def save_symbol_risk(symbol: str, stop_loss_money: float, take_profit_money: float) -> dict[str, float]:
    symbol = symbol.upper()
    if symbol not in _DEFAULTS:
        raise ValueError("market hanya BTCUSD atau XAUUSD")
    values = {
        "stop_loss_money": max(0.0, float(stop_loss_money)),
        "take_profit_money": max(0.0, float(take_profit_money)),
    }
    with _LOCK:
        _CONFIG[symbol] = values
        _persist()
    return values.copy()
