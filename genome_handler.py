import numpy as np
import random as rand
import math
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.layers.normalization import BatchNormalization

##################################
# Genomes are represented as fixed-with lists of integers corresponding
# to sequential layers and properties. A model with 2 recurrent layers
# and 1 dense layer would look like:
#
# [<recurr layer><recurr layer><dense layer><optimizer>]
#
# The makeup of the recurrent layers and dense layers is defined in
# the GenomeHandler below under self.recurr_layer_shape and
# self.dense_layer_shape. <optimizer> consists of just one property.
###################################

class GenomeHandler:
    def __init__(self, max_recurr_layers, max_dense_layers, max_recurr_nodes, max_dense_nodes, input_shape,
                 batch_normalization=True, dropout=True, optimizers=None, activations=None, initializers=None):
        if max_dense_layers < 1:
            raise ValueError("At least one dense layer is required for final sigmoid layer")
        self.optimizer = optimizers or [
            'sgd',
            'rmsprop',
            'adagrad',
            'adadelta',
            'adam',
            'adamax',
            'nadam'
        ]
        self.activation = activations or [
            'elu',
            'selu',
            'softplus',
            'softsign',
            'relu',
            'tanh',
            'sigmoid',
            'hard_sigmoid',
            'linear'
        ]
        self.initializer = initializers or [
            'zeros',
            'ones',
            'normal',
            'uniform',
            'truncated_normal',
            'orthogonal',
            'glorot_uniform'
        ]
        self.recurr_layer_shape = [
            "active",
            "num recurrent nodes",
            "batch normalization",
            "activation",
            "recurrent activation",
            "kernel initializer",
            "recurrent initializer",
            "dropout",
            "recurrent dropout",
        ]
        self.dense_layer_shape = [
            "active",
            "num dense nodes",
            "batch normalization",
            "activation",
            "kernel initializer",
        ]
        self.layer_params = {
            "active": [0, 1],
            "num recurrent nodes": [2 ** i for i in range(3, int(math.log(max_recurr_nodes, 2)) + 1)],
            "num dense nodes": [2**i for i in range(3, int(math.log(max_dense_nodes, 2)) + 1)],
            "batch normalization": [0, (1 if batch_normalization else 0)],
            "activation": list(range(len(self.activation))),
            "recurrent activation": list(range(len(self.activation))),
            "kernel initializer": list(range(len(self.initializer))),
            "recurrent initializer": list(range(len(self.initializer))),
            "dropout": [(i if dropout else 0) for i in range(11)],
            "recurrent dropout": [(i if dropout else 0) for i in range(11)],
        }

        self.recurr_layers = max_recurr_layers
        self.recurr_layer_size = len(self.recurr_layer_shape)
        self.dense_layers = max_dense_layers - 1 # this doesn't include the final sigmoid layer, so -1
        self.dense_layer_size = len(self.dense_layer_shape)
        self.input_shape = input_shape

    def recurrParam(self, i):
        key = self.recurr_layer_shape[i]
        return self.layer_params[key]

    def denseParam(self, i):
        key = self.dense_layer_shape[i]
        return self.layer_params[key]

    def mutate(self, genome, num_mutations):
        num_mutations = np.random.choice(num_mutations)
        for i in range(num_mutations):
            index = np.random.choice(list(range(1, len(genome))))

            if index < self.recurr_layer_size * self.recurr_layers:
                if genome[index - index % self.recurr_layer_size]:
                    range_index = index % self.recurr_layer_size
                    choice_range = self.recurrParam(range_index)
                    genome[index] = np.random.choice(choice_range)
                elif rand.uniform(0, 1) <= 0.01: # randomly flip deactivated layers
                    genome[index - index % self.recurr_layer_size] = 1
            elif index != len(genome) - 1:
                offset = self.recurr_layer_size * self.recurr_layers
                new_index = (index - offset)
                present_index = new_index - new_index % self.dense_layer_size
                if genome[present_index + offset]:
                    range_index = new_index % self.dense_layer_size
                    choice_range = self.denseParam(range_index)
                    genome[index] = np.random.choice(choice_range)
                elif rand.uniform(0, 1) <= 0.01:
                    genome[present_index + offset] = 1
            else:
                genome[index] = np.random.choice(list(range(len(self.optimizer))))
        return genome

    def decode(self, genome):
        if not self.is_compatible_genome(genome):
            raise ValueError("Invalid genome for specified configs")

        # I mark the last recurrent layer, which does not need to return sequences
        offset = 0
        for i in range(self.recurr_layers):
            if genome[offset]:
                last_recurr = offset
            offset += self.recurr_layer_size
        model = Sequential()
        offset = 0
        input_layer = True
        for i in range(self.recurr_layers):
            if genome[offset]:
                config = {'units': genome[offset + 1],
                          'activation': self.activation[genome[offset + 3]],
                          'recurrent_activation': self.activation[genome[offset + 4]],
                          'kernel_initializer': self.initializer[genome[offset + 5]],
                          'recurrent_initializer': self.initializer[genome[offset + 6]],
                          'dropout': float(genome[offset + 7] / 20.0),
                          'recurrent_dropout': float(genome[offset + 8] / 20.0)}
                if not last_recurr == offset:
                    config['return_sequences'] = True
                if input_layer:
                    config['batch_input_shape'] = (None,) + self.input_shape
                    recurr = LSTM.from_config(config)
                    input_layer = False
                else:
                    recurr = LSTM.from_config(config)
                model.add(recurr)
                if genome[offset + 2]:
                    model.add(BatchNormalization())
            offset += self.recurr_layer_size

        for i in range(self.dense_layers):
            if genome[offset]:
                config = {'units': genome[offset + 1],
                          'activation': self.activation[genome[offset + 3]],
                          'kernel_initializer': self.initializer[genome[offset + 4]]}
                if input_layer:
                    config['batch_input_shape'] = (None,) + self.input_shape
                    dense = Dense.from_config(config)
                    input_layer = False
                else:
                    dense = Dense.from_config(config)
                model.add(dense)
                if genome[offset + 2]:
                    model.add(BatchNormalization())
            offset += self.dense_layer_size

        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
            optimizer=self.optimizer[genome[offset]],
            metrics=["accuracy"])
        return model

    def genome_representation(self):
        encoding = []

        for i in range(self.recurr_layers):
            for key in self.recurr_layer_shape:
                encoding.append("Recurr" + str(i) + " " + key)
        for i in range(self.dense_layers):
            for key in self.dense_layer_shape:
                encoding.append("Dense" + str(i) + " " + key)
        encoding.append("Optimizer")
        return encoding

    def generate(self):
        genome = []
        for i in range(self.recurr_layers):
            for key in self.recurr_layer_shape:
                param = self.layer_params[key]
                genome.append(np.random.choice(param))
        for i in range(self.dense_layers):
            for key in self.dense_layer_shape:
                param = self.layer_params[key]
                genome.append(np.random.choice(param))
        genome.append(np.random.choice(list(range(len(self.optimizer)))))
        genome[0] = 1   # to ensure that at least one layer is active
        return genome

    def is_compatible_genome(self, genome):
        expected_len = self.recurr_layers * self.recurr_layer_size \
                       + self.dense_layers * self.dense_layer_size + 1
        if len(genome) != expected_len:
            return False
        ind = 0
        for i in range(self.recurr_layers):
            for j in range(self.recurr_layer_size):
                if genome[ind + j] not in self.recurrParam(j):
                    return False
            ind += self.recurr_layer_size
        for i in range(self.dense_layers):
            for j in range(self.dense_layer_size):
                if genome[ind + j] not in self.denseParam(j):
                    return False
            ind += self.dense_layer_size
        if genome[ind] not in range(len(self.optimizer)):
            return False
        return True

    # metric = accuracy or loss
    def best_genome(self, csv_path, metric="accuracy", include_metrics=True):
        best = max if metric is "accuracy" else min
        col = -1 if metric is "accuracy" else -2
        data = np.genfromtxt(csv_path, delimiter=",")
        row = list(data[:, col]).index(best(data[:, col]))
        genome = list(map(int, data[row, :-2]))
        if include_metrics:
            genome += list(data[row, -2:])
        return genome

    # metric = accuracy or loss
    def decode_best(self, csv_path, metric="accuracy"):
        return self.decode(self.best_genome(csv_path, metric, False))
