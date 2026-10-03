from datetime import datetime
import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

from .lstm_model import create_lstm_model, get_callbacks

MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "trained_models"
)
FEATURE_COLS = [
    "Close",
    "Volume",
    "High",
    "Low",
    "Returns",
    "MA_20",
    "MA_50",
    "Volatility",
]


def prepare_lstm_data(df: pd.DataFrame, sequence_length: int = 60):
    """Prepares scaled multi-feature vectors and 1-step ahead (t+1) targets."""
    missing_cols = [col for col in FEATURE_COLS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required feature columns: {missing_cols}")

    features_df = df[FEATURE_COLS].copy().dropna()
    if len(features_df) < sequence_length + 10:
        return None, None, None, None

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(features_df)

    X, y = [], []
    # Target is 1-step ahead Close price (index 0)
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i - sequence_length : i])
        y.append(scaled_data[i, 0])

    return np.array(X), np.array(y), features_df, scaler


def train_or_load_model(
    symbol: str, df: pd.DataFrame, sequence_length: int = 60
):
    """Loads cached ticker-specific model if trained within 7 days, else trains a new one."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    clean_symbol = (
        symbol.replace("=", "_")
        .replace("^", "_")
        .replace("-", "_")
        .replace(".", "_")
    )
    model_path = os.path.join(MODEL_DIR, f"{clean_symbol}_lstm.h5")
    scaler_path = os.path.join(MODEL_DIR, f"{clean_symbol}_scaler.pkl")

    # Check 7-day cache
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        mtime = datetime.fromtimestamp(os.path.getmtime(model_path))
        if (datetime.now() - mtime).days < 7:
            try:
                model = tf.keras.models.load_model(model_path)
                scaler = joblib.load(scaler_path)
                return model, scaler
            except Exception as e:
                print(f"Error loading cached model for {symbol}: {e}")

    # Train new model
    X, y, features_df, scaler = prepare_lstm_data(df, sequence_length)
    if X is None or len(X) < 50:
        return None, None

    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    model = create_lstm_model(input_shape=(X.shape[1], X.shape[2]))
    callbacks = get_callbacks()

    model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=40,
        batch_size=32,
        callbacks=callbacks,
        verbose=0,
    )

    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    return model, scaler


def predict_with_lstm(
    model,
    scaler,
    df: pd.DataFrame,
    days: int = 30,
    sequence_length: int = 60,
) -> list[float]:
    """Executes 1-step recursive rolling predictions for N horizon days."""
    features_df = df[FEATURE_COLS].copy().dropna()
    if len(features_df) < sequence_length:
        return []

    scaled_data = scaler.transform(features_df.iloc[-sequence_length:])
    current_seq = scaled_data.copy()
    predictions_unscaled = []

    for _ in range(days):
        input_tensor = np.expand_dims(current_seq, axis=0)
        pred_scaled_close = float(
            model.predict(input_tensor, verbose=0)[0, 0]
        )

        # Roll feature window forward
        last_row = current_seq[-1].copy()
        new_row = last_row.copy()
        new_row[0] = pred_scaled_close
        current_seq = np.vstack([current_seq[1:], new_row])

        # Unscale target price
        dummy_row = np.zeros((1, len(FEATURE_COLS)))
        dummy_row[0, 0] = pred_scaled_close
        unscaled_close = scaler.inverse_transform(dummy_row)[0, 0]
        predictions_unscaled.append(float(round(unscaled_close, 2)))

    return predictions_unscaled


def simple_prediction_fallback(
    df: pd.DataFrame, days: int = 30
) -> list[float]:
    """Monte Carlo drift and volatility simulation fallback for short historical series."""
    if df.empty or "Close" not in df.columns:
        return []

    last_price = float(df["Close"].iloc[-1])
    returns = df["Close"].pct_change().dropna()
    drift = float(returns.mean()) if not returns.empty else 0.0005
    volatility = float(returns.std()) if not returns.empty else 0.015

    predictions = []
    current_price = last_price
    np.random.seed(42)

    for _ in range(days):
        shock = np.random.normal(drift, volatility)
        current_price *= 1 + shock
        predictions.append(float(round(current_price, 2)))

    return predictions