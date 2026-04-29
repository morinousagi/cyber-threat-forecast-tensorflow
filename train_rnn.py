# train.py

import os
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from config import *
from data.preprocessing import load_and_clean_data, scale_data, create_sequences
from models.rnn_models import build_lstm_attention, build_gru_attention
from evaluation.metrics import evaluate_model

os.makedirs("artifacts", exist_ok=True)

# Load data
df = load_and_clean_data("cyber_data.csv")
scaled_data, scaler = scale_data(df)
X, y = create_sequences(scaled_data)

# TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=N_SPLITS)
for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

callbacks = [
    EarlyStopping(patience=PATIENCE_EARLY_STOP, restore_best_weights=True),
    ReduceLROnPlateau(patience=PATIENCE_LR, factor=0.5)
]

# Train LSTM
lstm = build_lstm_attention()
hist_lstm = lstm.fit(X_train, y_train,
                     validation_data=(X_test, y_test),
                     epochs=EPOCHS,
                     batch_size=BATCH_SIZE,
                     callbacks=callbacks,
                     verbose=1)

print("LSTM epochs:", len(hist_lstm.history["loss"]))

# Train GRU
gru = build_gru_attention()
hist_gru = gru.fit(X_train, y_train,
                   validation_data=(X_test, y_test),
                   epochs=EPOCHS,
                   batch_size=BATCH_SIZE,
                   callbacks=callbacks,
                   verbose=1)

print("GRU epochs:", len(hist_gru.history["loss"]))

# Evaluation
lstm_metrics = evaluate_model(lstm, X_test, y_test, scaler)
gru_metrics = evaluate_model(gru, X_test, y_test, scaler)

print("LSTM -> MSE:", lstm_metrics[0], "MAE:", lstm_metrics[1], "R2:", lstm_metrics[2])
print("GRU  -> MSE:", gru_metrics[0], "MAE:", gru_metrics[1], "R2:", gru_metrics[2])

# Plot Loss
plt.figure()
plt.plot(hist_lstm.history["loss"], label="LSTM")
plt.plot(hist_gru.history["loss"], label="GRU")
plt.legend()
plt.savefig("artifacts/rnn_training_loss.png")

# Plot Predictions
plt.figure()
plt.plot(lstm_metrics[4], label="Actual")
plt.plot(lstm_metrics[3], label="LSTM")
plt.plot(gru_metrics[3], label="GRU")
plt.legend()
plt.savefig("artifacts/rnn_predictions.png")

# Select Best Model
if lstm_metrics[0] < gru_metrics[0]:
    best_model = lstm
    print("Best Model: LSTM")
else:
    best_model = gru
    print("Best Model: GRU")

best_model.save("artifacts/best_rnn_model.keras")
joblib.dump(scaler, "artifacts/rnn_scaler.joblib")