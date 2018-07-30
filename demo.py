import numpy as np
import pandas as pd
import talib

from sklearn.preprocessing import MinMaxScaler
from geneticNN import DEvol, GenomeHandler


# **Prepare dataset**
# This problem uses historical financial data for the Reliance Industries Limited
# which have been downloaded from the Quandl database and saved in a csv file

# read dataset
dataset = pd.read_csv('Reliance.csv')

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

# define input
x = dataset[['10day SMA', '10day WMA', 'Momentum', 'Stochastic %K', 'Stochastic %D', 'RSI', 'MACD', 'Williams %R',
             'A/D Oscillator', 'CCI']].values
y = dataset['Price Rise'].values

# split into train and test sets
split = int(len(dataset)*0.8)
x_train, x_test, y_train, y_test = x[:split], x[split:], y[:split], y[split:]

# normalize features
scaler = MinMaxScaler(feature_range=(-1, 1))
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# **Prepare the genome configuration**
# The `GenomeHandler` class handles the constraints that are imposed upon
# models in a particular genetic program. See `genome-handler.py`
# for more information.

max_recurr_layers = 3
max_dense_layers = 4    # including final sigmoid layer
max_recurr_nodes = 512
max_dense_nodes = 1024

# reshape input to be 3D [samples, timesteps, features]
if max_recurr_layers != 0:
    x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
    x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))

dataset = ((x_train, y_train), (x_test, y_test))
input_shape = x_train.shape[1:]

genome_handler = GenomeHandler(max_recurr_layers, max_dense_layers, max_recurr_nodes, max_dense_nodes, input_shape)

# **Create and run the genetic program**
# The next, and final, step is create a `DEvol` and run it. Here we specify
# a few settings pertaining to the genetic program. The program
# will save each genome's encoding, as well as the model's loss and
# accuracy, in a `.csv` file printed at the beginning of program.

num_generations = 40
population_size = 40
num_epochs = 10000

devol = DEvol(genome_handler, 'genomes.csv')
model = devol.run(dataset, num_generations, population_size, num_epochs, metric='accuracy')
model.summary()
