"""Pydantic request/response schemas for the API."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class StatusResponse(BaseModel):
    running: bool
    trading_enabled: bool
    mt5_connected: bool
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
    max_open_positions: int = 3
    auto_symbols: list[str] = Field(default_factory=list)


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
    symbol: Literal["BTCUSD", "XAUUSD"]


class BulkLevelRequest(BaseModel):
    level: Literal["SL", "TP"]
    price: float = Field(gt=0.0)


class MT5LoginRequest(BaseModel):
    login: int = Field(gt=0)
    password: str = Field(min_length=1)
    server: str = Field(min_length=1)


class AutoMarketsRequest(BaseModel):
    symbols: list[Literal["BTCUSD", "XAUUSD"]] = Field(min_length=1, max_length=2)
