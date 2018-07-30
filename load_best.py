import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import model_from_json
from sklearn.preprocessing import MinMaxScaler

# valuta la possibilita di scrivere una funzione che restituisce x_train, y_train, x_test, y_test opportunamente scalati
# da usare qui e in demo.py
dataset = pd.read_csv('WaltDisney.csv')

dataset['Tomorrow Close'] = dataset['Close'].shift(-1)
dataset = dataset.dropna()

x = dataset['Close'].values.reshape(-1, 1)
y = dataset['Tomorrow Close'].values.reshape(-1, 1)

split = int(len(dataset)*0.8)
x_train, x_test, y_train, y_test = x[:split], x[split:], y[:split], y[split:]

scaler = MinMaxScaler(feature_range=(0, 1))
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
y_test = scaler.transform(y_test)

step_size = 1
x_train = np.reshape(x_train, (x_train.shape[0], step_size, x_train.shape[1]))
x_test = np.reshape(x_test, (x_test.shape[0], step_size, x_test.shape[1]))
y_test = np.reshape(y_test, y_test.size)

with open('best-model.json') as file:
    model = model_from_json(file.read())
model.load_weights('best-model.h5')

model.compile(loss='mean_squared_error', optimizer='adagrad')
score = model.evaluate(x_test, y_test)
print('loss = {}'.format(score))

y_predict_train = scaler.inverse_transform(model.predict(x_train))
y_predict_test = scaler.inverse_transform(model.predict(x_test))

start = len(y_predict_train)
end = len(y)

plt.plot(y, label='correct answers')
plt.plot(y_predict_train, label='prediction on training set')
plt.plot(range(start, end), y_predict_test, label='prediction on testing set')
plt.legend(loc = 'upper right')
plt.show()