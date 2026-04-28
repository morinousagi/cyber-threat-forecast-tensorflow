# data/preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from config import WINDOW_SIZE, TARGET_COLUMN, DATE_COLUMN

def load_and_clean_data(path):
    df = pd.read_csv(path)

    df[DATE_COLUMN] = pd.to_datetime(
        df[DATE_COLUMN],
        format='%d/%m/%Y %H:%M',
        errors='coerce'
    )

    df = df.dropna()
    df = df.sort_values(DATE_COLUMN)

    # Country-level aggregation
    df_daily = df.groupby([DATE_COLUMN, 'Country'])[TARGET_COLUMN].mean().reset_index()
    df_global = df_daily.groupby(DATE_COLUMN)[TARGET_COLUMN].mean().reset_index()

    df_global.set_index(DATE_COLUMN, inplace=True)

    return df_global


def scale_data(df):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    return scaled, scaler


def create_sequences(data, window=WINDOW_SIZE):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window])
        y.append(data[i+window])
    return np.array(X), np.array(y)