from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

import talib
import numpy as np

# read dataset
dataset = read_csv('Reliance.csv')

# Drop data older than 2008-01-01
mask = (dataset['Date'] > '2008-01-01')
dataset = dataset.loc[mask]

dataset['10day SMA'] = dataset['Close'].shift(1).rolling(window = 10).mean()
dataset['10day WMA'] = talib.WMA(dataset['Close'].values, timeperiod=10)
dataset['Momentum'] = talib.MOM(dataset['Close'].values, timeperiod=9)
dataset['Stochastic %K'], dataset['Stochastic %D'] = talib.STOCH(dataset['High'].values, dataset['Low'].values,
                                                                 dataset['Close'].values)
dataset['RSI'] = talib.RSI(dataset['Close'].values, timeperiod = 9)
dataset['MACD'] = talib.MACD(dataset['Close'].values)[0]
dataset['Williams %R'] = talib.WILLR(dataset['High'].values, dataset['Low'].values, dataset['Close'].values, 7)
dataset['A/D Oscillator'] = talib.ADOSC(dataset['High'].values, dataset['Low'].values, dataset['Close'].values,
                                        dataset['Total Trade Quantity'].values)
dataset['CCI'] = talib.CCI(dataset['High'].values, dataset['Low'].values, dataset['Close'].values)
dataset['Price Rise'] = np.where(dataset['Close'].shift(-1) > dataset['Close'], 1, 0)
dataset = dataset.dropna()

days = dataset[['10day SMA', '10day WMA', 'Momentum', 'Stochastic %K', 'Stochastic %D', 'RSI', 'MACD', 'Williams %R',
                'A/D Oscillator', 'CCI']].values

# normalize features
scaler = MinMaxScaler(feature_range=(-1, 1))
days = scaler.fit_transform(days)

# define input
step_size = 14
x = np.empty((days.shape[0] - step_size + 1, step_size, days.shape[1]))
for i in range(len(days) - step_size + 1):
    x[i] = days[i:step_size + i]
y = dataset['Price Rise'].values[step_size - 1:]

# split into train and test sets
split = int(len(dataset)*0.8)
x_train, x_test, y_train, y_test = x[:split], x[split:], y[:split], y[split:]

# # reshape input to be 3D [samples, timesteps, features]
# x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
# x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))

# design network
model = Sequential()
model.add(LSTM(10, input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(Dense(50, activation='tanh'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

# fit network
model.fit(x_train, y_train, epochs=5000, validation_data=(x_test, y_test), verbose=1, shuffle=False)