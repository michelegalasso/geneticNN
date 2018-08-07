import random as rand


class Population(object):

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
