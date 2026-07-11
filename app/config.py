"""Application configuration loaded from environment / .env file.

Uses pydantic-settings so every value is validated and type-cast.
Nothing here is hardcoded with credentials - all sensitive values come
from the environment.
"""
from __future__ import annotations

from functools import lru_cache
import json
import os
from pathlib import Path
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly-typed application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Instrument ---
    symbol: str = Field(default="XAUUSD")
    # NoDecode: skip pydantic-settings' JSON pre-parse so the validator below
    # can accept a plain comma-separated string from the .env file.
    timeframes: Annotated[list[str], NoDecode] = Field(default=["M1", "M5", "M15", "H1"])
    candles: int = Field(default=500)

    # --- Risk / lot ---
    lot_default: float = Field(default=0.01)
    risk_percent: float = Field(default=1.0)
    risk_reward: float = Field(default=2.0)
    max_spread_points: int = Field(default=300)
    max_open_positions: int = Field(default=3, ge=1)
    btcusd_min_lot: float = Field(default=0.05, gt=0.0)
    xauusd_min_lot: float = Field(default=0.01, gt=0.0)

    # --- Behaviour / safety ---
    trading_enabled: bool = Field(default=False)
    auto_start: bool = Field(default=False)
    target_profit_money: float = Field(default=5.0)
    trailing_stop: bool = Field(default=True)
    trailing_profit_step_money: float = Field(default=1.0, ge=0.0)
    trailing_start_points: int = Field(default=200)
    trailing_step_points: int = Field(default=100)
    magic_number: int = Field(default=220619)

    # --- ML ---
    ml_prob_threshold: float = Field(default=0.70)
    model_path: str = Field(default="app/ml/models/xgboost_model.json")
    lightgbm_model_path: str = Field(default="app/ml/models/lightgbm_model.joblib")
    xgboost_ensemble_weight: float = Field(default=0.5, ge=0.0, le=1.0)

    # --- Multi-timeframe confirmation ---
    mtf_confirmation: bool = Field(default=True)
    mtf_min_agreement: float = Field(default=0.5, ge=0.0, le=1.0)

    # --- API ---
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)

    # --- Logging ---
    log_level: str = Field(default="INFO")

    def model_post_init(self, __context: object) -> None:
        """Apply the default market assigned to this account in accounts.json."""
        config_path = Path(os.getenv("ACCOUNTS_CONFIG_PATH", "settings/accounts.json"))
        account_id = os.getenv("ACCOUNT_ID", "default").strip().lower() or "default"
        try:
            accounts = json.loads(config_path.read_text(encoding="utf-8")).get("accounts", [])
            profile = next(item for item in accounts if str(item.get("account_id", "")).lower() == account_id)
            configured_symbol = str(profile.get("symbol", "")).strip().upper()
            if configured_symbol:
                self.symbol = configured_symbol
        except (FileNotFoundError, StopIteration, TypeError, ValueError, json.JSONDecodeError):
            pass

    @field_validator("timeframes", mode="before")
    @classmethod
    def _split_timeframes(cls, v: object) -> list[str]:
        """Allow TIMEFRAMES to be a comma-separated string in the .env file."""
        if isinstance(v, str):
            return [item.strip().upper() for item in v.split(",") if item.strip()]
        return v  # type: ignore[return-value]

    @field_validator("symbol")
    @classmethod
    def _upper_symbol(cls, v: str) -> str:
        return v.strip()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance (singleton)."""
    return Settings()


# Convenient module-level instance.
settings = get_settings()
