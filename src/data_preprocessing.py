import numpy as np
from sklearn.preprocessing import MinMaxScaler


def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data["Close"].values.reshape(-1, 1))

    # Create sequences for time windows
    X, y = [], []
    time_window = 60
    for i in range(len(data) - time_window):
        X.append(data_scaled[i : i + time_window])
        y.append(data_scaled[i + time_window])
    return np.array(X), np.array(y), scaler
