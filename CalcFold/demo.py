import numpy as np
import pandas as pd

from geneticNN.GeneticAlgo import GeneticAlgo
from geneticNN.GenomeHandler import GenomeHandler
from sklearn.preprocessing import MinMaxScaler


# This example uses HP's data. Feel free to experiment with other data.
# But while doing so, be careful to have a large enough dataset and also pay attention to the data normalization

filename = 'hpq.us.csv'     # csv file in which data are stored
df = pd.read_csv(filename, delimiter=',', usecols=['Date','Open','High','Low','Close'])
print('Loaded data from {}'.format(filename))

df = df.sort_values('Date')

# First calculate the mid prices from the highest and lowest
high_prices = df.loc[:,'High'].values
low_prices = df.loc[:,'Low'].values
mid_prices = (high_prices+low_prices)/2.0

train_data = mid_prices[:11000]
test_data = mid_prices[11000:]

# Scale the data to be between 0 and 1
# When scaling remember! You normalize both test and train data with respect to training data
# Because you are not supposed to have access to test data
scaler = MinMaxScaler()
train_data = train_data.reshape(-1,1)
test_data = test_data.reshape(-1,1)

# Train the Scaler with training data and smooth data
smoothing_window_size = 2500
for di in range(0,10000,smoothing_window_size):
    scaler.fit(train_data[di:di+smoothing_window_size,:])
    train_data[di:di+smoothing_window_size,:] = scaler.transform(train_data[di:di+smoothing_window_size,:])

# You normalize the last bit of remaining data
scaler.fit(train_data[di+smoothing_window_size:,:])
train_data[di+smoothing_window_size:,:] = scaler.transform(train_data[di+smoothing_window_size:,:])

# Reshape both train and test data
train_data = train_data.reshape(-1)

# Normalize test data
test_data = scaler.transform(test_data).reshape(-1)

# Now perform exponential moving average smoothing
# So the data will have a smoother curve than the original ragged data
EMA = 0.0
gamma = 0.1
for ti in range(11000):
  EMA = gamma*train_data[ti] + (1-gamma)*EMA
  train_data[ti] = EMA

# **Prepare the genome configuration**
# The `GenomeHandler` class handles the constraints that are imposed upon
# models in a particular genetic program. See `genome-handler.py`
# for more information.

max_layers = 4
max_nodes = 200     # maximum number of nodes in each layer
epochs_range = [5, 15]

genome_handler = GenomeHandler(max_layers, max_nodes, epochs_range)

# **Create and run the genetic program**
# The next, and final, step is create a `DEvol` and run it. Here we specify
# a few settings pertaining to the genetic program. The program
# will save each genome's encoding, as well as the model's loss and
# accuracy, in a `.csv` file printed at the beginning of program.

num_generations = 20
population_size = 20

genetic = GeneticAlgo(genome_handler)
best_model = genetic.run(train_data, test_data, num_generations, population_size)

# print the prediction of the best model
best_model.show_results(df, np.concatenate([train_data,test_data], axis=0))
