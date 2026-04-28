# evaluation/metrics.py

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def evaluate_model(model, X_test, y_test, scaler):
    preds = model.predict(X_test)

    preds_inv = scaler.inverse_transform(preds)
    y_test_inv = scaler.inverse_transform(y_test)

    mse = mean_squared_error(y_test_inv, preds_inv)
    mae = mean_absolute_error(y_test_inv, preds_inv)
    r2 = r2_score(y_test_inv, preds_inv)

    return mse, mae, r2, preds_inv, y_test_inv