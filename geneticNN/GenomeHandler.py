import numpy as np
import random as rand

from geneticNN.CustomLSTM import CustomLSTM

##################################
# Genomes are represented as fixed-with lists of integers corresponding to
# sequential layers and properties. A model with 3 layers would look like:
#
# [<ID><layer 1><layer 2><layer 3>]
###################################


class GenomeHandler(object):
    def __init__(self, max_layers, max_nodes, epochs_range):
        '''
        Initializes the parameters used to build genomes.

        :param max_layers (int): maximum number of layers in the neural network
        :param max_nodes (int): maximum number of nodes in each layer
        :param epochs_range (list): minimum and maximum value of epochs to train the neural network
        '''
        self.layer_shape = [
            "active",
            "nodes",
        ]
        self.layer_params = {
            "active": [0, 1],
            "nodes": [i for i in range(10, max_nodes + 1, 10)],
        }
        self.epochs = list(range(epochs_range[0], epochs_range[1] + 1))

        self.layers = max_layers
        self.layer_size = len(self.layer_shape)
        self.ID = 0

    def Param(self, i):
        key = self.layer_shape[i]
        return self.layer_params[key]

    def mutate(self, genome, num_mutations):
        num_mutations = np.random.choice(num_mutations)
        new_genome = genome[1:]
        for i in range(num_mutations):
            index = np.random.choice(list(range(1, len(new_genome))))

            if index < self.layer_size * self.layers:
                if new_genome[index - index % self.layer_size]:
                    range_index = index % self.layer_size
                    choice_range = self.Param(range_index)
                    new_genome[index] = np.random.choice(choice_range)
                elif rand.uniform(0, 1) <= 0.01: # randomly flip deactivated layers
                    new_genome[index - index % self.layer_size] = 1
            else:
                new_genome[index] = np.random.choice(self.epochs)

        return [genome[0]] + new_genome

    def decode(self, genome):
        if not self.is_compatible_genome(genome):
            raise ValueError("Invalid genome for specified configs")

        ID = genome[0]
        offset = 1
        num_nodes = []

        for layer in range(self.layers):
            if genome[offset]:
                num_nodes.append(genome[offset + 1])
            offset += self.layer_size

        model = CustomLSTM(ID, num_nodes)
        epochs = genome[offset]
        return model, epochs

    def genome_representation(self):
        encoding = ['ID']
        for i in range(self.layers):
            for key in self.layer_shape:
                encoding.append('Layer' + str(i + 1) + " " + key)
        encoding.append('Epochs')
        return encoding

    def generate(self, ID):
        genome = [ID]
        for i in range(self.layers):
            for key in self.layer_shape:
                param = self.layer_params[key]
                genome.append(np.random.choice(param))
        genome.append(np.random.choice(self.epochs))
        genome[1] = 1   # to ensure that at least one layer is active
        return genome

    def is_compatible_genome(self, genome):
        expected_len = self.layers * self.layer_size + 2
        if len(genome) != expected_len:
            return False
        ind = 1
        for i in range(self.layers):
            for j in range(self.layer_size):
                if genome[ind + j] not in self.Param(j):
                    return False
            ind += self.layer_size
        if genome[ind] not in self.epochs:
            return False
        return True
