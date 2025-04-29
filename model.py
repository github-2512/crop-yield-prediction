import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import joblib

X = np.random.rand(500, 4)
y = (X[:, 0] * 0.3 + X[:, 1] * 0.2 + X[:, 2] * 0.4 + X[:, 3] * 0.1) * 100

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

model = Sequential()
model.add(Dense(64, input_dim=4, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit(X_scaled, y, epochs=50, batch_size=16)

model.save('model.h5')
joblib.dump(scaler, 'scaler.save')

print("✅ Model trained and saved!")