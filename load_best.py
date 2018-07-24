import talib
import numpy as np
import pickle

from keras.models import model_from_json
from sklearn.preprocessing import StandardScaler

with open('ABB_India.pickle', 'rb') as file:
    dataset = pickle.load(file)

dataset = dataset.dropna()
dataset = dataset[['Open', 'High', 'Low', 'Close']]

dataset['H-L'] = dataset['High'] - dataset['Low']
dataset['O-C'] = dataset['Close'] - dataset['Open']
dataset['3day MA'] = dataset['Close'].shift(1).rolling(window = 3).mean()
dataset['10day MA'] = dataset['Close'].shift(1).rolling(window = 10).mean()
dataset['30day MA'] = dataset['Close'].shift(1).rolling(window = 30).mean()
dataset['Std_dev']= dataset['Close'].rolling(5).std()
dataset['RSI'] = talib.RSI(dataset['Close'].values, timeperiod = 9)
dataset['Williams %R'] = talib.WILLR(dataset['High'].values, dataset['Low'].values, dataset['Close'].values, 7)
dataset['Price_Rise'] = np.where(dataset['Close'].shift(-1) > dataset['Close'], 1, 0)

dataset = dataset.dropna()

x = dataset.iloc[:, 4:-1]
y = dataset.iloc[:, -1]

split = int(len(dataset)*0.8)
x_train, x_test, y_train, y_test = x[:split], x[split:], y[:split], y[split:]

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

y_train = y_train.values
y_test = y_test.values

with open('best-model.json', 'r') as json_file:
    model = model_from_json(json_file.read())

model.load_weights('best-model.h5')
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
y_pred = model.predict(x_test)
y_pred = np.rint(y_pred.reshape(y_pred.shape[0])).astype(int)

counter = 0
for prediction, value in zip(y_pred, y_test):
    if prediction == value:
        counter += 1
print('Model accuracy: {}'.format(counter / len(y_pred)))