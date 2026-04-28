# Cyber Threat Forecasting using ARIMA, LSTM & GRU with Attention

This projected is completed using **prompt engineering with ChatGPT**.
```
this project uses dataset 'cyber_data.csv' as described in https://github.com/DrSufi/CyberData.
this is time-series data with over 77K rows and 18 fields. to be sorted by 'AttackDate'. 
'AttackDate' is of obj dtype - need to convert using pd.to_datetime(df['AttackDate'], format='%d/%m/%Y %H:%M', errors='coerce').
drop rows with missing values.
analyze country-level trends and forecast future cyber threat levels.
sliding window of 14 days to forecast future threats, for example predict 'Local Infection'
use time series split with n_splits =5 or you can recommend a value
use tf.keras to create models 2-layer LSTM and 2-layer GRU with attention, or better.
include early stopping callback and learning rate scheduler
use adam for optimizer.
print the total number of epochs for each model.
use metrics MSE MAE and R-squared and print the values for each model.
plot training loss of both models on the same plot.
plot model predictions of both models on the same plot.
compare model performance, select the better performing model and save best model in .joblib and .h5 files.
for deployment to Streamlit Cloud.
generate all codes/scripts, requirements.txt and README.md

first 3 rows of data looks like this: ...
```

## 📌 Overview

This project forecasts future cyber threat levels using time-series deep learning models.

Target variable: `Local Infection`

Implement and compare:
- 2-Layer LSTM with Attention
- 2-Layer GRU with Attention

The best performing model is automatically selected and saved for deployment to Streamlit Cloud.

---

## 📊 Dataset Description

- Source: https://github.com/DrSufi/CyberData
- Citation: Sufi, F. A New Time Series Dataset for Cyber-Threat Correlation, Regression and Neural-Network-Based Forecasting. Information 2024, 15, 199. https://doi.org/10.3390/info15040199

The dataset contains:

- 77,000+ rows
- 18 features
- Country-level cyber threat indicators
- Timestamp column: `AttackDate`


Data preprocessing steps:

- Convert `AttackDate` to datetime
- Sort by time
- Drop missing values
- Aggregate daily country-level threat
- Global aggregation across countries
- Normalize using MinMaxScaler

---

## 🔁 Forecasting Strategy

- Sliding window: 14 days
- Target: Next day Local Infection
- TimeSeriesSplit cross-validation (n_splits = 5)
- Adam optimizer
- EarlyStopping
- ReduceLROnPlateau

---

## 🧠 Model Architectures

### LSTM Model

- LSTM (64 units)
- Dropout
- LSTM (32 units)
- Attention Layer
- GlobalAveragePooling
- Dense Output

### GRU Model

- GRU (64 units)
- Dropout
- GRU (32 units)
- Attention Layer
- GlobalAveragePooling
- Dense Output

---

## 📏 Evaluation Metrics

Each model is evaluated using:

- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- R-squared (R²)

Training loss comparison plot is generated.

Prediction comparison plot is generated.

Best model is selected based on lowest MSE.

## Results

GRU vs LSTM results - both has negative R2 (model performs worse than simply predicting the mean of the target.)

```
Epoch 1/100
9/9 ━━━━━━━━━━━━━━━━━━━━ 2s 50ms/step - loss: 0.2316 - val_loss: 0.0947 - learning_rate: 0.0010
Epoch 2/100
9/9 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step - loss: 0.0832 - val_loss: 0.0587 - learning_rate: 0.0010
Epoch 3/100
9/9 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step - loss: 0.0636 - val_loss: 0.0700 - learning_rate: 0.0010
Epoch 4/100
9/9 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step - loss: 0.0661 - val_loss: 0.0543 - learning_rate: 0.0010
Epoch 5/100
9/9 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step - loss: 0.0605 - val_loss: 0.0528 - learning_rate: 0.0010
Epoch 6/100
9/9 ━━━━━━━━━━━━━━━━━━━━ 0s 18ms/step - loss: 0.0610 - val_loss: 0.0548 - learning_rate: 0.0010
Epoch 7/100
9/9 ━━━━━━━━━━━━━━━━━━━━ 0s 25ms/step - loss: 0.0596 - val_loss: 0.0532 - learning_rate: 0.0010
Epoch 8/100
9/9 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step - loss: 0.0602 - val_loss: 0.0534 - learning_rate: 0.0010
Epoch 9/100
9/9 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step - loss: 0.0604 - val_loss: 0.0539 - learning_rate: 0.0010
Epoch 10/100
9/9 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step - loss: 0.0601 - val_loss: 0.0539 - learning_rate: 0.0010
Epoch 11/100
9/9 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step - loss: 0.0601 - val_loss: 0.0535 - learning_rate: 5.0000e-04
Epoch 12/100
9/9 ━━━━━━━━━━━━━━━━━━━━ 0s 18ms/step - loss: 0.0598 - val_loss: 0.0541 - learning_rate: 5.0000e-04
Epoch 13/100
9/9 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step - loss: 0.0599 - val_loss: 0.0533 - learning_rate: 5.0000e-04
Epoch 14/100
9/9 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step - loss: 0.0599 - val_loss: 0.0539 - learning_rate: 5.0000e-04
Epoch 15/100
9/9 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step - loss: 0.0608 - val_loss: 0.0544 - learning_rate: 5.0000e-04
LSTM epochs: 15
Epoch 1/100
9/9 ━━━━━━━━━━━━━━━━━━━━ 3s 78ms/step - loss: 0.1956 - val_loss: 0.0544 - learning_rate: 0.0010
Epoch 2/100
9/9 ━━━━━━━━━━━━━━━━━━━━ 0s 22ms/step - loss: 0.0811 - val_loss: 0.0570 - learning_rate: 0.0010
Epoch 3/100
9/9 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step - loss: 0.0620 - val_loss: 0.0653 - learning_rate: 0.0010
Epoch 4/100
9/9 ━━━━━━━━━━━━━━━━━━━━ 0s 24ms/step - loss: 0.0660 - val_loss: 0.0579 - learning_rate: 0.0010
Epoch 5/100
9/9 ━━━━━━━━━━━━━━━━━━━━ 0s 23ms/step - loss: 0.0620 - val_loss: 0.0539 - learning_rate: 0.0010
Epoch 6/100
9/9 ━━━━━━━━━━━━━━━━━━━━ 0s 21ms/step - loss: 0.0619 - val_loss: 0.0545 - learning_rate: 5.0000e-04
Epoch 7/100
9/9 ━━━━━━━━━━━━━━━━━━━━ 0s 25ms/step - loss: 0.0614 - val_loss: 0.0561 - learning_rate: 5.0000e-04
Epoch 8/100
9/9 ━━━━━━━━━━━━━━━━━━━━ 0s 23ms/step - loss: 0.0620 - val_loss: 0.0551 - learning_rate: 5.0000e-04
Epoch 9/100
9/9 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step - loss: 0.0610 - val_loss: 0.0547 - learning_rate: 5.0000e-04
Epoch 10/100
9/9 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step - loss: 0.0613 - val_loss: 0.0548 - learning_rate: 5.0000e-04
GRU epochs: 10
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 193ms/step
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 224ms/step
LSTM -> MSE: 7.183661652382978e-06 MAE: 0.0023180479463759205 R2: -0.003101294491351103
GRU  -> MSE: 7.403461033878675e-06 MAE: 0.0024720920214856617 R2: -0.03379330850534523
Best Model: LSTM
```

---

## 📂 Project Structure
```
/
├── app.py
├── train.py
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
│   ├── best_model.h5
│   ├── scaler.joblib
│   ├── training_loss.png
│   └── predictions.png
└── cyber_data.csv
```