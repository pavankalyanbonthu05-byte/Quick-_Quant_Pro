from datetime import datetime
from typing import Any, Dict

from .fetcher import fetch_stock_data
from .predictor import (
    predict_with_lstm,
    simple_prediction_fallback,
    train_or_load_model,
)
from .schemas import QuantSignal, TechnicalIndicators


def predict_stock_trend(symbol: str, days: int = 30) -> Dict[str, Any]:
    """Main tool entrypoint for Orchestrator Agent.

    Fetches market history, computes technical indicators, executes 1-step
    recursive LSTM predictions (or Monte Carlo fallback), and returns a
    structured JSON-compatible signal dictionary.
    """
    symbol = symbol.strip().upper()
    df, latest_stats = fetch_stock_data(symbol)

    if df.empty or not latest_stats:
        raise ValueError(
            f"Unable to fetch valid market data for ticker '{symbol}'."
        )

    current_price = latest_stats["Close"]
    is_fallback = False

    # Execute LSTM pipeline with automatic fallback on failure
    try:
        model, scaler = train_or_load_model(symbol, df)
        if model is not None and scaler is not None:
            trajectory = predict_with_lstm(model, scaler, df, days=days)
        else:
            trajectory = simple_prediction_fallback(df, days=days)
            is_fallback = True
    except Exception as e:
        print(
            f"[Worker 1 Warning] LSTM model failed for {symbol}, switching to fallback: {e}"
        )
        trajectory = simple_prediction_fallback(df, days=days)
        is_fallback = True

    if not trajectory:
        trajectory = [current_price] * days
        is_fallback = True

    target_price = trajectory[-1]
    return_pct = round(((target_price - current_price) / current_price) * 100, 2)

    tech_indicators = TechnicalIndicators(
        rsi_14=round(latest_stats["RSI"], 2),
        ma_20=round(latest_stats["MA20"], 2),
        ma_50=round(latest_stats["MA50"], 2),
        volatility_20d=round(latest_stats["Volatility"], 4),
        returns_pct=round(latest_stats["ChangePercent"], 2),
        volume_ratio=round(latest_stats["VolumeRatio"], 2),
    )

    signal = QuantSignal(
        symbol=symbol,
        as_of_date=latest_stats["Date"],
        current_price=round(current_price, 2),
        forecast_horizon_days=days,
        predicted_target_price=round(target_price, 2),
        forecasted_return_pct=return_pct,
        forecast_trajectory=trajectory,
        technical_indicators=tech_indicators,
        is_fallback=is_fallback,
    )

    return signal.model_dump()