"""Pydantic request/response schemas for the API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class StatusResponse(BaseModel):
    running: bool
    trading_enabled: bool
    mt5_connected: bool
    symbol: str
    timeframes: list[str]
    open_positions: int
    model_loaded: bool


class AccountResponse(BaseModel):
    connected: bool
    info: dict | None = None


class PositionsResponse(BaseModel):
    count: int
    total_profit: float
    positions: list[dict]


class SignalResponse(BaseModel):
    timeframe: str
    signal: dict


class TrainRequest(BaseModel):
    csv: str = Field(..., description="Path to OHLCV CSV file")
    horizon: int = 1
    atr_mult: float = 0.5
    test_size: float = 0.2


class BacktestRequest(BaseModel):
    csv: str = Field(..., description="Path to OHLCV CSV file")
    start_balance: float = 1000.0
    warmup: int = 200
    max_hold: int = 96


class ActionResponse(BaseModel):
    ok: bool
    message: str
    detail: dict | list | None = None
