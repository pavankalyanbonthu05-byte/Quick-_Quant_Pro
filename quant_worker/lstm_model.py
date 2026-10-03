import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential


def create_lstm_model(input_shape: tuple[int, int]) -> Sequential:
    """Builds a 3-layer stacked LSTM network with Dropout layers.

    input_shape: (sequence_length, feature_count) -> e.g., (60, 8)
    """
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(100, return_sequences=True),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25, activation="relu"),
        Dense(1),
    ])
    model.compile(
        optimizer="adam", loss="mean_squared_error", metrics=["mae"]
    )
    return model


def get_callbacks() -> list:
    """Returns training callbacks for early stopping and learning rate adaptation."""
    return [
        EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5
        ),
    ]