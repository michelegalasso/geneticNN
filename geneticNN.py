import numpy as np
from datetime import datetime
import random as rand
import csv
import operator
import os


class Genetic:

    def __init__(self, genome_handler, data_path=""):
        self.genome_handler = genome_handler
        self.datafile = data_path or (datetime.now().ctime() + '.csv')
        self.bssf = -1
        self.best_model = None
        self.ID = 0

        if os.path.isfile(data_path) and os.stat(data_path).st_size > 1:
            raise ValueError('Non-empty file %s already exists. Please change file path to prevent overwritten genome data.' % data_path)

        print("Genome encoding and accuracy data stored at", self.datafile, "\n")
        with open(self.datafile, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            genome = genome_handler.genome_representation() + ["Val Loss"]
            writer.writerow(genome)

    def set_objective(self):
        self.metric = 'loss'
        self.objective = "min"
        self.metric_op = operator.__lt__
        self.metric_objective = min


    def run(self, train_data, test_data, num_generations, pop_size, epochs, scoring_function=None, frac_crossover=0.8):
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
        self.train_data = train_data
        self.test_data = test_data
        self.set_objective()

        # Generate initial random population
        members = []
        for _ in range(pop_size):
            self.ID += 1
            members.append(self.genome_handler.generate(self.ID))

        fit = []
        for i in range(len(members)):
            print("\nmodel {0}/{1} - generation {2}/{3}:\n"\
                    .format(i + 1, len(members), 1, num_generations))
            res = self.evaluate(members[i], epochs)
            fit.append(res)

        fit = np.array(fit)
        pop = Population(members, fit, scoring_function, obj=self.objective)
        print("Generation {3}:\t\tbest {4}: {0:0.5f}\t\taverage: {1:0.5f}\t\tstd: {2:0.5f}"\
                .format(self.metric_objective(fit), np.mean(fit), np.std(fit), 1, self.metric))

        # Evolve over 
        for gen in range(1, num_generations):
            best_genome, best_fit = pop.getBest()  # genome and fitness of the best individual of the previous population
            members = []
            for i in range(int((pop_size - 1)*frac_crossover)):  # Crossover
                members.append(self.crossover(pop.select(), pop.select()))
            for i in range(len(members)):  # Mutation
                members[i] = self.mutate(members[i], gen)

            # new random individuals
            for _ in range(pop_size - int((pop_size - 1) * frac_crossover) - 1):
                self.ID += 1
                members.append(self.genome_handler.generate(self.ID))

            fit = []
            for i in range(len(members)):
                print("\nmodel {0}/{1} - generation {2}/{3}:\n"
                        .format(i + 1, len(members), gen + 1, num_generations))
                res = self.evaluate(members[i], epochs)
                fit.append(res)
            members.append(best_genome)
            fit.append(best_fit)

            fit = np.array(fit)
            pop = Population(members, fit, scoring_function, obj=self.objective)
            print("Generation {3}:\t\tbest {4}: {0:0.5f}\t\taverage: {1:0.5f}\t\tstd: {2:0.5f}"\
                    .format(self.metric_objective(pop.fitnesses), np.mean(pop.fitnesses), np.std(pop.fitnesses),
                            gen + 1, self.metric))

        return self.best_model

    def evaluate(self, genome, epochs):
        model = self.genome_handler.decode(genome)
        loss = model.train(self.train_data, self.test_data, epochs=epochs)

        # Record the stats
        with open(self.datafile, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',',
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)
            row = list(genome) + [loss]
            writer.writerow(row)

        met = loss
        if self.bssf is -1 or self.metric_op(met, self.bssf):
            self.bssf = met
            self.best_model = model
            print('\nNew best individual found: {}\n'.format(genome))

        return loss

    def crossover(self, genome1, genome2):
        self.ID += 1
        crossIndexA = rand.randint(0, len(genome1))
        child = genome1[:crossIndexA] + genome2[crossIndexA:]
        child[0] = self.ID
        return child

    def mutate(self, genome, generation):
        # increase mutations as program continues
        num_mutations = max(3, generation // 4)
        return self.genome_handler.mutate(genome, num_mutations)


class Population:

    def __len__(self):
        return len(self.members)

    def __init__(self, members, fitnesses, scoring_function, obj='min'):
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
