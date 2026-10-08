import numpy as np
import pandas as pd


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def fetch_stock_data(symbol: str) -> tuple[pd.DataFrame, dict]:
    symbol = symbol.strip().upper()
    from .predictor import fetch_ohlcv_direct
    df = fetch_ohlcv_direct(symbol)

    if df.empty or "Close" not in df.columns or len(df) < 30:
        return pd.DataFrame(), {}

    # Technical Indicators
    df["Returns"] = df["Close"].pct_change()
    df["MA_20"] = df["Close"].rolling(window=20).mean()
    df["MA_50"] = df["Close"].rolling(window=50).mean()
    df["Volatility"] = df["Returns"].rolling(window=20).std()
    df["RSI"] = calculate_rsi(df["Close"], period=14)

    avg_vol_20 = df["Volume"].rolling(window=20).mean()
    df["Volume_Ratio"] = df["Volume"] / (avg_vol_20 + 1e-10)

    rolling_252 = min(len(df), 252)
    high_52w = df["High"].iloc[-rolling_252:].max()
    low_52w = df["Low"].iloc[-rolling_252:].min()

    df = df.ffill().bfill().fillna(0)

    latest = df.iloc[-1]
    prev_close = df.iloc[-2]["Close"] if len(df) > 1 else latest["Close"]
    price_change = latest["Close"] - prev_close
    pct_change = (price_change / prev_close) * 100 if prev_close != 0 else 0.0

    today_stats = {
        "Date": (
            latest.name.strftime("%Y-%m-%d")
            if hasattr(latest.name, "strftime")
            else str(latest.name)
        ),
        "Open": float(latest["Open"]),
        "High": float(latest["High"]),
        "Low": float(latest["Low"]),
        "Close": float(latest["Close"]),
        "Volume": float(latest["Volume"]),
        "Change": float(price_change),
        "ChangePercent": float(pct_change),
        "RSI": float(latest["RSI"]),
        "MA20": float(latest["MA_20"]),
        "MA50": float(latest["MA_50"]),
        "Volatility": float(latest["Volatility"]),
        "VolumeRatio": float(latest["Volume_Ratio"]),
        "52w_high": float(high_52w),
        "52w_low": float(low_52w),
    }

    return df, today_stats