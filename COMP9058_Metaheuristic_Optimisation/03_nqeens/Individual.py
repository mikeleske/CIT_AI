"""
Author: Mike Leske
file: Individual.py

Use GA to solve the n-queens problem
"""

import random
import math


class Individual:
    def __init__(self, _n, _genes=None):
        """
        Parameters and general variables
        """
        self.genSize    = _n
        self.genes      = _genes
        self.fitness    = 0
        self.cost       = 0

        if not _genes:
            self.genes = [i for i in range(1, self.genSize+1)]
            random.shuffle(self.genes)

        self.computeFitness()

    def copy(self):
        """
        Creating a new individual
        """
        ind = Individual(self.genSize)
        for i in range(0, self.genSize):
            ind.genes[i] = self.genes[i]
        ind.fitness = self.getFitness()
        ind.cost = self.getCost()
        return ind

    def computeFitness(self):
        """
        Computing the cost or fitness of the individual
        """
        for i in range(self.genSize):
            for j in range(self.genSize):
                if ( i != j):
                    dx = abs(i-j)
                    dy = abs(self.genes[i] - self.genes[j])
                    if(dx == dy):
                        self.cost += 1
        
        if self.cost != 0:
            self.fitness = 1/self.cost 
        else: 
            self.fitness = 1
            
    def getFitness(self):
        return self.fitness

    def getCost(self):
        return self.cost
