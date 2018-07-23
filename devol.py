from genome_handler import GenomeHandler
import numpy as np
from keras.models import Sequential
from keras.utils import np_utils
from keras.datasets import mnist, cifar10
from keras.callbacks import EarlyStopping
from keras.models import load_model
import keras.backend as K
import tensorflow as tf
from datetime import datetime
import random as rand
import csv
import sys
import operator
import gc
import os

METRIC_OPS = [operator.__lt__, operator.__gt__]
METRIC_OBJECTIVES = [min, max]


class DEvol:

    def __init__(self, genome_handler, data_path=""):
        self.genome_handler = genome_handler
        self.datafile = data_path or (datetime.now().ctime() + '.csv')
        self.bssf = -1

        if os.path.isfile(data_path) and os.stat(data_path).st_size > 1:
            raise ValueError('Non-empty file %s already exists. Please change file path to prevent overwritten genome data.' % data_path)

        print("Genome encoding and accuracy data stored at", self.datafile, "\n")
        with open(self.datafile, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            genome = genome_handler.genome_representation() + ["Val Loss", "Val Accuracy"]
            writer.writerow(genome)

    def set_objective(self, metric):
        """set the metric and objective for this search  should be 'accuracy' or 'loss'"""
        if metric is 'acc':
            metric = 'accuracy'
        if not metric in ['loss', 'accuracy']:
            raise ValueError(
                'Invalid metric name {} provided - should be "accuracy" or "loss"'.format(metric))
        self.metric = metric
        self.objective = "max" if self.metric is "accuracy" else "min"
        self.metric_index = 1 if self.metric is 'loss' else -1
        self.metric_op = METRIC_OPS[self.objective is 'max']
        self.metric_objective = METRIC_OBJECTIVES[self.objective is 'max']


    def run(self, dataset, num_generations, pop_size, epochs, scoring_function=None, metric='accuracy',
            frac_crossover=0.8):
        """run genetic search on dataset given number of generations and population size

        Args:
            dataset : tuple or list of numpy arrays in form ((train_data, train_labels), (validation_data, validation_labels))
            num_generations (int): number of generations to search
            pop_size (int): initial population size
            epochs (int): epochs to run each search, passed to keras model.fit -currently searches are
                            curtailed if no improvement is seen in 1 epoch
            scoring_function (None, optional): scoring function to be applied to population scores, will be called on a numpy array
                                      which is a  min/max scaled version of evaluated model metrics, so
                                      It should accept a real number including 0. If left as default just the min/max
                                      scaled values will be used.
            metric (str, optional): must be "accuracy" or "loss" , defines what to optimize during search

        Returns:
            keras model: best model found
        """
        self.set_objective(metric)
        (self.x_train, self.y_train), (self.x_test, self.y_test) = dataset
        # Generate initial random population
        members = [self.genome_handler.generate() for _ in range(pop_size)]
        fit = []
        metric_index = 1 if self.metric is 'loss' else -1
        for i in range(len(members)):
            print("\nmodel {0}/{1} - generation {2}/{3}:\n"\
                    .format(i + 1, len(members), 1, num_generations))
            res = self.evaluate(members[i], epochs)
            v = res[metric_index]
            del res
            fit.append(v)

        fit = np.array(fit)
        pop = Population(members, fit, scoring_function, obj=self.objective)
        print("Generation {3}:\t\tbest {4}: {0:0.4f}\t\taverage: {1:0.4f}\t\tstd: {2:0.4f}"\
                .format(self.metric_objective(fit), np.mean(fit), np.std(fit), 1, self.metric))

        # Evolve over 
        for gen in range(1, num_generations):
            best_genome, best_fit = pop.getBest()  # genome and fitness of the best individual of the previous population
            members = []
            for i in range(int((pop_size - 1)*frac_crossover)):  # Crossover
                members.append(self.crossover(pop.select(), pop.select()))
            for i in range(len(members)):  # Mutation
                members[i] = self.mutate(members[i], gen)
            new_random_individuals = [self.genome_handler.generate()
                                      for _ in range(pop_size - int((pop_size - 1)*frac_crossover) - 1)]
            members.extend(new_random_individuals)
            fit = []
            for i in range(len(members)):
                print("\nmodel {0}/{1} - generation {2}/{3}:\n"
                        .format(i + 1, len(members), gen + 1, num_generations))
                res = self.evaluate(members[i], epochs)
                v = res[metric_index]
                del res
                fit.append(v)
            members.append(best_genome)
            fit.append(best_fit)

            fit = np.array(fit)
            pop = Population(members, fit, scoring_function, obj=self.objective)
            print("Generation {3}:\t\tbest {4}: {0:0.4f}\t\taverage: {1:0.4f}\t\tstd: {2:0.4f}"\
                    .format(self.metric_objective(pop.fitnesses), np.mean(pop.fitnesses), np.std(pop.fitnesses),
                            gen + 1, self.metric))

        return load_model('best-model.h5')

    def evaluate(self, genome, epochs):
        model = self.genome_handler.decode(genome)
        loss, accuracy = None, None
        try:
            model.fit(self.x_train, self.y_train, validation_data=(self.x_test, self.y_test),
                      epochs=epochs,
                      verbose=1,
                      callbacks=[EarlyStopping(monitor='val_loss', patience=5, verbose=1)])
            loss, accuracy = model.evaluate(self.x_test, self.y_test, verbose=0)
        except:
            loss = 6.66
            accuracy = 1 / self.genome_handler.n_classes
            gc.collect()
            K.clear_session()
            tf.reset_default_graph()
            print("An error occurred and the model could not train. Assigned poor score.")
        # Record the stats
        with open(self.datafile, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',',
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)
            row = list(genome) + [loss, accuracy]
            writer.writerow(row)

        met = loss if self.metric == 'loss' else accuracy
        if self.bssf is -1 or self.metric_op(met, self.bssf) and accuracy is not 0:
            try:
                os.remove('best-model.json')
                os.remove('best-model.h5')
            except OSError:
                pass
            self.bssf = met
            with open('best-model.json', 'w') as json_file:
                json_file.write(model.to_json())
            model.save('best-model.h5')

        return model, loss, accuracy

    def crossover(self, genome1, genome2):
        crossIndexA = rand.randint(0, len(genome1))
        child = genome1[:crossIndexA] + genome2[crossIndexA:]
        return child

    def mutate(self, genome, generation):
        # increase mutations as program continues
        num_mutations = max(3, generation // 4)
        return self.genome_handler.mutate(genome, num_mutations)


class Population:

    def __len__(self):
        return len(self.members)

    def __init__(self, members, fitnesses, scoring_function, obj='max'):
        self.members = members
        self.fitnesses = fitnesses
        scores = fitnesses - fitnesses.min()
        if scores.max() > 0:
            scores /= scores.max()
        if obj is 'min':
            scores = 1 - scores
        if scoring_function:
            self.scores = scoring_function(scores)
        else:
            self.scores = scores
        self.s_scores = sum(self.scores)

    def getBest(self):
        combined = [(self.members[i], self.fitnesses[i])
                    for i in range(len(self.members))]
        combined = sorted(combined, key=(lambda x: x[1]), reverse=True)
        return combined[0]     # returns genome and fit of the best individual

    def select(self):
        dart = rand.uniform(0, self.s_scores)
        sum_scores = 0
        for i in range(len(self.members)):
            sum_scores += self.scores[i]
            if sum_scores >= dart:
                return self.members[i]
