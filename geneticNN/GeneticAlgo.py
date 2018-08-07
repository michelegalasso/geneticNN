import numpy as np
import random as rand
import operator
import os

from geneticNN.Population import Population


class GeneticAlgo(object):

    def __init__(self, genome_handler):
        self.genome_handler = genome_handler
        self.bssf = -1
        self.best_model = None
        self.ID = 0
        self.keys = genome_handler.genome_representation() + ['Val Loss']
        self.lengths = [len(word) + 4 for word in self.keys]

        # check for previous results
        if os.path.isfile('Individuals.txt') or os.path.isfile('BestIndividuals.txt'):
            raise ValueError('Previous results found. Please remove them or move them in a different path.')

        # write output
        with open('Individuals.txt', 'a') as file:
            file.write(''.join(item.ljust(length) for item, length in zip(self.keys, self.lengths)))
            file.write('\n')

        with open('BestIndividuals.txt', 'a') as file:
            file.write(''.join(item.ljust(length) for item, length in zip(self.keys, self.lengths)))
            file.write('\n')

    def set_objective(self):
        self.metric = 'loss'
        self.objective = "min"
        self.metric_op = operator.__lt__
        self.metric_objective = min


    def run(self, train_data, test_data, num_generations, pop_size, scoring_function=None, frac_crossover=0.8):
        """run genetic search on dataset given number of generations and population size

        Args:
            dataset : tuple or list of numpy arrays in form ((train_data, train_labels), (validation_data, validation_labels))
            num_generations (int): number of generations to search
            pop_size (int): initial population size
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
            res = self.evaluate(members[i])
            fit.append(res)

        fit = np.array(fit)
        pop = Population(members, fit, scoring_function, obj=self.objective)
        print("Generation {3}:\t\tbest {4}: {0:0.5f}\t\taverage: {1:0.5f}\t\tstd: {2:0.5f}"\
                .format(self.metric_objective(fit), np.mean(fit), np.std(fit), 1, self.metric))

        # Evolve over 
        for gen in range(1, num_generations):
            best_genome, best_fit = pop.getBest()  # genome and fitness of the best individual of the previous population

            # write output
            values = best_genome + [best_fit]
            with open('BestIndividuals.txt', 'a') as file:
                file.write('Generation {}\n'.format(gen))
                file.write(''.join(str(value).ljust(length) for value, length in zip(values, self.lengths)))
                file.write('\n')

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
                res = self.evaluate(members[i])
                fit.append(res)
            members.append(best_genome)
            fit.append(best_fit)

            fit = np.array(fit)
            pop = Population(members, fit, scoring_function, obj=self.objective)
            print("Generation {3}:\t\tbest {4}: {0:0.5f}\t\taverage: {1:0.5f}\t\tstd: {2:0.5f}"\
                    .format(self.metric_objective(pop.fitnesses), np.mean(pop.fitnesses), np.std(pop.fitnesses),
                            gen + 1, self.metric))

        # write output for the last generation
        best_genome, best_fit = pop.getBest()
        values = best_genome + [best_fit]
        with open('BestIndividuals.txt', 'a') as file:
            file.write('Generation {}\n'.format(num_generations))
            file.write(''.join(str(value).ljust(length) for value, length in zip(values, self.lengths)))
            file.write('\n')

        return self.best_model

    def evaluate(self, genome):
        model, epochs = self.genome_handler.decode(genome)
        loss = model.train(self.train_data, self.test_data, epochs=epochs)
        values = genome + [loss]

        # write output
        with open('Individuals.txt', 'a') as file:
            file.write(''.join(str(value).ljust(length) for value, length in zip(values, self.lengths)))
            file.write('\n')

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
