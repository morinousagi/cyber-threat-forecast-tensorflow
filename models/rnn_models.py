# models/rnn_models.py

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Input, Attention, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from config import WINDOW_SIZE, LEARNING_RATE


def build_lstm_attention():
    inputs = Input(shape=(WINDOW_SIZE, 1))
    x = LSTM(64, return_sequences=True)(inputs)
    x = Dropout(0.2)(x)
    x = LSTM(32, return_sequences=True)(x)
    attn = Attention()([x, x])
    x = GlobalAveragePooling1D()(attn)
    output = Dense(1)(x)

    model = Model(inputs, output)
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="mse"
    )
    return model


def build_gru_attention():
    inputs = Input(shape=(WINDOW_SIZE, 1))
    x = GRU(64, return_sequences=True)(inputs)
    x = Dropout(0.2)(x)
    x = GRU(32, return_sequences=True)(x)
    attn = Attention()([x, x])
    x = GlobalAveragePooling1D()(attn)
    output = Dense(1)(x)

    model = Model(inputs, output)
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="mse"
    )
    return model