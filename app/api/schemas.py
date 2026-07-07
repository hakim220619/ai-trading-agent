"""Pydantic request/response schemas for the API."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class StatusResponse(BaseModel):
    running: bool
    trading_enabled: bool
    mt5_connected: bool
    broker_connected: bool = False
    connection_stable: bool = False
    ping_ms: float | None = None
    ping_jitter_ms: float | None = None
    tick_age_seconds: float | None = None
    symbol: str
    timeframes: list[str]
    open_positions: int
    model_loaded: bool
    terminal_trade_allowed: bool
    account_trade_allowed: bool
    trade_api_disabled: bool
    account_balance: float | None = None
    account_equity: float | None = None
    account_currency: str | None = None
    total_profit: float = 0.0
    confidence_auto: bool = False
    active_strategy: Literal["confidence_m5", "recovery_m1"] = "confidence_m5"
    strategy_status: dict = Field(default_factory=dict)
    max_open_positions: int = 3
    auto_symbols: list[str] = Field(default_factory=list)
    confidence_threshold: float = 0.65
    trailing_stop: bool = True
    trailing_profit_step_money: float = 1.0
    daily_profit_limit_enabled: bool = False
    daily_profit_limit_money: float = 0.0
    daily_lot_limit_enabled: bool = False
    daily_lot_limit: float = 0.0
    daily_profit_today: float = 0.0
    daily_lot_today: float = 0.0


class AccountResponse(BaseModel):
    connected: bool
    info: dict | None = None


class PositionsResponse(BaseModel):
    count: int
    total_profit: float
    positions: list[dict]


class TradeHistoryResponse(BaseModel):
    days: int
    summary: dict
    deals: list[dict]


class SignalResponse(BaseModel):
    timeframe: str
    signal: dict
    trade_plan: dict | None = None


class TrainRequest(BaseModel):
    csv: str = Field(..., description="Path to OHLCV CSV file")
    horizon: int = 1
    atr_mult: float = 0.5
    test_size: float = 0.2


class ExportRequest(BaseModel):
    symbol: str | None = None
    timeframe: str = "M5"
    count: int = Field(default=10_000, ge=200, le=100_000)


class BacktestRequest(BaseModel):
    csv: str = Field(..., description="Path to OHLCV CSV file")
    start_balance: float = 1000.0
    warmup: int = 200
    max_hold: int = 96
    signal_lookback: int = Field(default=500, ge=50, le=10_000)
    account_profile: str = "exness-pro"
    use_historical_spread: bool = True
    commission_per_lot_side: float = Field(default=0.0, ge=0.0)
    slippage_points: float = Field(default=0.0, ge=0.0)


class ActionResponse(BaseModel):
    ok: bool
    message: str
    detail: dict | list | None = None


class ManualTradeRequest(BaseModel):
    direction: Literal["BUY", "SELL"]
    lot: float = Field(default=0.01, gt=0.0, le=100.0)


class SymbolRequest(BaseModel):
    symbol: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9._\-#]+$")


class BulkLevelRequest(BaseModel):
    level: Literal["SL", "TP"]
    price: float = Field(gt=0.0)


class MT5LoginRequest(BaseModel):
    login: int = Field(gt=0)
    password: str = ""
    server: str = Field(min_length=1)
    account_id: str | None = Field(default=None, min_length=1, max_length=40, pattern=r"^[A-Za-z0-9_-]+$")
    label: str | None = Field(default=None, max_length=80)
    terminal_path: str | None = Field(default=None, max_length=500)


class AutoMarketsRequest(BaseModel):
    symbols: list[str] = Field(min_length=1, max_length=20)


class SymbolRiskConfigRequest(BaseModel):
    symbol: Literal["BTCUSD", "XAUUSD"]
    stop_loss_money: float = Field(ge=0.0, le=1_000_000.0)
    take_profit_money: float = Field(ge=0.0, le=1_000_000.0)


class TradingSetupRequest(BaseModel):
    active_strategy: Literal["confidence_m5", "recovery_m1"] | None = None
    confidence_threshold: float = Field(ge=0.5, le=0.99)
    max_open_positions: int = Field(ge=1, le=20)
    btcusd_min_lot: float = Field(ge=0.01, le=100.0)
    xauusd_min_lot: float = Field(ge=0.01, le=100.0)
    trailing_stop: bool = True
    trailing_profit_step_money: float = Field(gt=0.0)
    daily_profit_limit_enabled: bool = False
    daily_profit_limit_money: float = Field(default=0.0, ge=0.0, le=1_000_000.0)
    daily_lot_limit_enabled: bool = False
    daily_lot_limit: float = Field(default=0.0, ge=0.0, le=1_000.0)


class ScalpingSetupRequest(BaseModel):
    symbol: Literal["BTCUSD", "XAUUSD"]
    confidence_threshold: float = Field(ge=0.50, le=0.99)
    base_lot: float = Field(ge=0.01, le=100.0)
    second_lot: float = Field(ge=0.01, le=100.0)
    lot_multiplier: float = Field(ge=1.0, le=10.0)
    max_lot: float = Field(ge=0.01, le=100.0)
    initial_loss_money: float = Field(gt=0.0, le=1_000_000.0)
    loss_increment_money: float = Field(gt=0.0, le=1_000_000.0)
    basket_profit_target: float = Field(gt=0.0, le=1_000_000.0)
    basket_loss_limit: float = Field(default=0.0, ge=0.0, le=1_000_000.0)
    basket_loss_limit_enabled: bool = False
    daily_profit_target: float = Field(default=0.0, ge=0.0, le=1_000_000.0)
    daily_profit_target_enabled: bool = False
    daily_loss_limit: float = Field(default=0.0, ge=0.0, le=1_000_000.0)
    daily_loss_limit_enabled: bool = False
