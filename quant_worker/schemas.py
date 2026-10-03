from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class TechnicalIndicators(BaseModel):
    rsi_14: float = Field(..., description="14-period Relative Strength Index")
    ma_20: float = Field(..., description="20-day Simple Moving Average")
    ma_50: float = Field(..., description="50-day Simple Moving Average")
    volatility_20d: float = Field(..., description="20-day rolling daily volatility")
    returns_pct: float = Field(..., description="Latest daily percentage return")
    volume_ratio: float = Field(
        ..., description="Ratio of current volume to 20-day average volume"
    )


class QuantSignal(BaseModel):
    agent_id: str = "worker1_lstm_forecaster"
    symbol: str
    as_of_date: str
    current_price: float
    forecast_horizon_days: int = 30
    predicted_target_price: float
    forecasted_return_pct: float
    forecast_trajectory: List[float]
    technical_indicators: TechnicalIndicators
    is_fallback: bool = False
    model_architecture: str = "3-Layer Stacked LSTM (100-100-50) + Dropout"