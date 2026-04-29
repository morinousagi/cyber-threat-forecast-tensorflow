# train_regression.py

import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

# =============================
# 1. Load Data
# =============================

df = pd.read_csv("cyber_data.csv")

df['AttackDate'] = pd.to_datetime(
    df['AttackDate'],
    format='%d/%m/%Y %H:%M',
    errors='coerce'
)

df = df.dropna()
df = df.sort_values("AttackDate")

# =============================
# 2. Aggregate Daily Global
# =============================

features = [
    'Spam',
    'Ransomware',
    'Exploit',
    'Malicious Mail',
    'Network Attack',
    'On Demand Scan',
    'Web Threat'
]

target = "Local Infection"

df_daily = df.groupby("AttackDate")[features + [target]].mean().reset_index()
df_daily.set_index("AttackDate", inplace=True)

# =============================
# 3. Create Lag Features
# =============================

WINDOW = 14

for col in features + [target]:
    for lag in range(1, WINDOW+1):
        df_daily[f"{col}_lag_{lag}"] = df_daily[col].shift(lag)

df_daily = df_daily.dropna()

X = df_daily.drop(columns=[target])
y = df_daily[target]

# =============================
# 4. Time Series Split
# =============================

tscv = TimeSeriesSplit(n_splits=5)

mse_scores = []
mae_scores = []
r2_scores = []

for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    print(f"\nFold {fold+1}")

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print("MSE:", mse)
    print("MAE:", mae)
    print("R2 :", r2)

    mse_scores.append(mse)
    mae_scores.append(mae)
    r2_scores.append(r2)

# =============================
# 5. Average Metrics
# =============================

print("\nAverage Performance")
print("MSE:", np.mean(mse_scores))
print("MAE:", np.mean(mae_scores))
print("R2 :", np.mean(r2_scores))

# =============================
# 6. Train Final Model on Full Data
# =============================

final_model = XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

final_model.fit(X, y)

os.makedirs("artifacts", exist_ok=True)
joblib.dump(final_model, "artifacts/best_regression_model.joblib")

print("\nModel saved to artifacts/best_regression_model.joblib")

# =============================
# 7. Final Evaluation Plots
# =============================

print("\nGenerating evaluation plots...")

# Use last fold for visualization
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

final_model.fit(X_train, y_train)
preds = final_model.predict(X_test)

# Ensure artifacts directory exists
os.makedirs("artifacts", exist_ok=True)

# -----------------------------
# 1. Actual vs Predicted
# -----------------------------
plt.figure(figsize=(10,5))
plt.plot(y_test.values, label="Actual")
plt.plot(preds, label="Predicted")
plt.title("Actual vs Predicted - XGBoost")
plt.legend()
plt.tight_layout()
plt.savefig("artifacts/xgb_actual_vs_predicted.png")
plt.close()

# -----------------------------
# 2. Residuals Plot
# -----------------------------
residuals = y_test.values - preds

plt.figure(figsize=(8,5))
plt.scatter(preds, residuals)
plt.axhline(0)
plt.title("Residuals vs Predicted")
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.tight_layout()
plt.savefig("artifacts/xgb_residuals.png")
plt.close()

# -----------------------------
# 3. Residual Distribution
# -----------------------------
plt.figure(figsize=(8,5))
plt.hist(residuals, bins=30)
plt.title("Residual Distribution")
plt.tight_layout()
plt.savefig("artifacts/xgb_residual_distribution.png")
plt.close()

# -----------------------------
# 4. Feature Importance
# -----------------------------
importance = final_model.feature_importances_
indices = np.argsort(importance)[::-1][:20]  # top 20 features

plt.figure(figsize=(10,6))
plt.bar(range(len(indices)), importance[indices])
plt.xticks(range(len(indices)),
           X.columns[indices],
           rotation=90)
plt.title("Top 20 Feature Importances")
plt.tight_layout()
plt.savefig("artifacts/xgb_feature_importance.png")
plt.close()

print("All evaluation plots saved in artifacts/")