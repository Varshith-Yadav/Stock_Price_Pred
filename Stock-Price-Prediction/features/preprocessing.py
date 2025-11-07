import numpy as np
from sklearn.preprocessing import MinMaxScaler

def scale_data(df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(df)
    return scaled, scaler

def create_sequences(data, window=60):
    x, y = [], []
    for i in range(window, len(data)):
        x.append(data[i-window:i])
        y.append(data[i, 3])  # Close price column
    return np.array(x), np.array(y)
