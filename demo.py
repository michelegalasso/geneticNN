import quandl
import talib
import numpy as np

from sklearn.preprocessing import StandardScaler
from devol import DEvol, GenomeHandler


# **Prepare dataset**
# This problem uses the Quandl financial database in order to get historical
# data for the company ABB India.

quandl.ApiConfig.api_key = '6f5z9_5XwN55jrrTuqaN'   # my key, to be removed later

dataset = quandl.get('NSE/ABB')
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

dataset = ((x_train, y_train), (x_test, y_test))

# **Prepare the genome configuration**
# The `GenomeHandler` class handles the constraints that are imposed upon
# models in a particular genetic program. See `genome-handler.py`
# for more information.

max_dense_layers = 4    # including final sigmoid layer
max_dense_nodes = 1024
input_shape = x_train.shape[1:]

genome_handler = GenomeHandler(max_dense_layers, max_dense_nodes, input_shape)

# **Create and run the genetic program**
# The next, and final, step is create a `DEvol` and run it. Here we specify
# a few settings pertaining to the genetic program. The program
# will save each genome's encoding, as well as the model's loss and
# accuracy, in a `.csv` file printed at the beginning of program.

num_generations = 40
population_size = 20
num_epochs = 100

devol = DEvol(genome_handler, 'genomes.csv')
model = devol.run(dataset, num_generations, population_size, num_epochs)
model.summary()
