# Cyber Threat Forecasting using XGBoost, LSTM & GRU with Attention

This projected is completed using **prompt engineering with ChatGPT**. 

Demo app deployed to Streamlit Cloud [link](https://morinousagi-cyber-threat-forecast-rnn-xgb.streamlit.app)

## Overview

This project builds a production-ready machine learning system to forecast cyber threat levels using time-series data.

The goal is to predict **Local Infection rates** using historical cyber threat indicators such as:
- Spam
- Ransomware
- Exploit
- Network Attack
- Web Threat
- Malicious Mail
- On Demand Scan

The system compares deep learning (RNN) approaches and tree-based regression models, and selects the best performing model.

Final solution uses **XGBoost regression with lag-based time-series features**, achieving strong predictive performance.

---

## 📊 Dataset

- Source: https://github.com/DrSufi/CyberData
- Citation: Sufi, F. A New Time Series Dataset for Cyber-Threat Correlation, Regression and Neural-Network-Based Forecasting. Information 2024, 15, 199. https://doi.org/10.3390/info15040199

Dataset characteristics:

- 77,000+ rows
- 18 features
- Country-level cyber indicators
- Timestamp column: `AttackDate`

Target variable:
`Local Infection`

---

## Methodology

### 1. Data Preprocessing

- Convert `AttackDate` to datetime
- Sort chronologically
- Drop missing values
- Aggregate daily averages
- Generate 14-day lag features
- Preserve temporal ordering

---

### 2. Feature Engineering

For each threat metric:

- Create 14 lag features
- Use multivariate inputs
- TimeSeriesSplit cross-validation

---

### 3. Models Compared

- 2-layer LSTM with Attention
- 2-layer GRU with Attention
- XGBoost Regression (final selected model)

Tree-based regression significantly outperformed RNN models on this structured dataset.

---

## Evaluation Strategy

Time-series aware validation using: TimeSeriesSplit (n_splits = 5)

Metrics:

- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- R² Score

Additional evaluation:

- Actual vs Predicted plot
- Residual analysis (XGBoost)
- Residual distribution (XGBoost)
- Feature importance ranking (XGBoost)

---

## Results

The XGBoost model achieved:
- Strong positive R²
- Stable residual distribution
- Meaningful feature importance patterns

Tree-based regression proved more suitable than RNN for this medium-sized structured dataset.

Both GRU & LSTM produced **negative R²**.

```
LSTM epochs: 15
GRU epochs: 10
LSTM -> MSE: 7.183661652382978e-06 MAE: 0.0023180479463759205 R2: -0.003101294491351103
GRU  -> MSE: 7.403461033878675e-06 MAE: 0.0024720920214856617 R2: -0.03379330850534523

XGBoost Average Performance
MSE: 1.052007685459743e-06
MAE: 0.0006856177082767775
R2 : 0.8864993006707829

Best Model: XGBoost
```

---

## Visual Outputs

Generated evaluation artifacts:

- Actual vs Predicted trend comparison
- Residual scatter plot
- Residual distribution histogram
- Top 20 feature importance plot

---

## 📂 Project Structure
```
/
├── app.py
├── train_rnn.py
├── train_regression.py
├── config.py
├── requirements.txt
├── README.md
├── models/
│   ├── __init__.py
│   └── rnn_models.py
├── data/
│   ├── __init__.py
│   └── preprocessing.py
├── evaluation/
│   ├── __init__.py
│   └── metrics.py
├── artifacts/
│   ├── best_regression_model.joblib
│   ├── best_rnn_model.keras
│   ├── rnn*
│   └── xgb*
└── cyber_data.csv
```