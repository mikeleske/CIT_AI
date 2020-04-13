"""
Author: Mike Leske
file: nqueens.py

Use GA to solve the n-queens problem
"""

import random
from Individual import *
import sys
import numpy as np

from datetime import datetime
import statistics
import argparse

class GA:
    def __init__(self, _n, _configuration, _popSize, _mutationRate, _maxIterations, _offsprings, _elite):
        """
        Parameters and general variables
        """
        self.genSize        = _n
        self.population     = []
        self.matingPool     = []
        self.best           = None
        self.popSize        = _popSize
        self.mutationRate   = _mutationRate
        self.maxIterations  = _maxIterations
        self.iteration      = 0
        self.data           = {}

        self.configuration  = _configuration
        self.offsprings     = _offsprings
        self.elite          = _elite

        self.initPopulation()

    def initPopulation(self):
        """
        Creating random individuals in the population
        """
        for _ in range(0, self.popSize):
            individual = Individual(self.genSize)
            self.population.append(individual)
            self.updateBest(individual)

        print ("Best initial sol: ", self.best.getFitness())

    def updateBest(self, candidate):
        if self.best == None or candidate.getFitness() > self.best.getFitness():
            self.best = candidate.copy()
            print ("  >> iteration:", self.iteration, "best:", self.best.getFitness(), "- violations:", self.best.getCost())
            return True

        if self.best.getFitness() == 1:
            print('Solution found:', self.best.genes)
            exit()

    def randomSelection(self):
        """
        Random (uniform) selection of two individuals
        """
        indA = self.matingPool[ random.randint(0, self.popSize-1) ]
        indB = self.matingPool[ random.randint(0, self.popSize-1) ]
        return [indA, indB]

    def stochasticUniversalSampling(self):
        """
        Your stochastic universal sampling Selection Implementation

        Algorithm implemented based on pseudocode from:
        http://ml.informatik.uni-freiburg.de/former/_media/teaching/ss13/ml/riedmiller/evolution.pdf - Slide 41
        """

        totalFitness = 0
        for ind in self.population:
            totalFitness += ind.getFitness()

        weightedFitness = [ ind.getFitness()/totalFitness for ind in self.population ]
        weightedPool = []

        poolSize = self.popSize
        r = random.uniform(0, 1/poolSize)
        idx = 0

        while len(weightedPool) < poolSize:
            while r <= weightedFitness[idx]:
                weightedPool.append(self.population[idx])
                r += 1/poolSize
            r -= weightedFitness[idx]
            idx += 1
        
        indA = weightedPool[ random.randint(0, poolSize-1) ]
        indB = weightedPool[ random.randint(0, poolSize-1) ]

        return [indA, indB]    

    def uniformCrossover(self, indA, indB):
        """
        Your Uniform Crossover Implementation
        """
        childA = []
        childB = []

        tmpA = indA.genes.copy()
        tmpB = indB.genes.copy()

        mask = [round(random.random()) for i in range(self.genSize)]

        if self.offsprings == 1:
            for x in range(len(mask)):
                if mask[x]:
                    childA.append(indA.genes[x])
                    tmpB.remove(indA.genes[x])
                else:
                    childA.append(None)

            for x in range(len(mask)):
                if not mask[x]:
                    childA[x] = tmpB.pop(0)
        else:
            for x in range(len(mask)):
                if mask[x]:
                    childA.append(indA.genes[x])
                    childB.append(indB.genes[x])
                    tmpA.remove(indB.genes[x])
                    tmpB.remove(indA.genes[x])
                else:
                    childA.append(None)
                    childB.append(None)

            for x in range(len(mask)):
                if not mask[x]:
                    childA[x] = tmpB.pop(0)
                    childB[x] = tmpA.pop(0)

        return childA, childB

    def pmxCrossover(self, indA, indB):
        """
        Your PMX Crossover Implementation
        """
        childA = [None] * self.genSize
        childB = [None] * self.genSize

        idx1, idx2 = sorted(random.sample(range(0, self.genSize), 2))
        mapA = childA[idx1:idx2+1] = indB.genes[idx1:idx2+1].copy()
        mapB = childB[idx1:idx2+1] = indA.genes[idx1:idx2+1].copy()

        def geneMapping(child, gene, m1, m2):
            mapping = m2[m1.index(gene)]
            while mapping in child:
                mapping = geneMapping(child, mapping, m1, m2)
            return mapping

        if self.offsprings == 1:
            for idx in range(len(childA)):
                if not childA[idx]:
                    if not indA.genes[idx] in childA:
                        childA[idx] = indA.genes[idx]
                    else:
                        childA[idx] = geneMapping(childA, indA.genes[idx], mapA, mapB)
        else:
            for idx in range(len(childA)):
                if not childA[idx]:
                    if not indA.genes[idx] in childA:
                        childA[idx] = indA.genes[idx]
                    else:
                        childA[idx] = geneMapping(childA, indA.genes[idx], mapA, mapB)
                    
                    if not indB.genes[idx] in childB:
                        childB[idx] = indB.genes[idx]
                    else:
                        childB[idx] = geneMapping(childB, indB.genes[idx], mapB, mapA)
        
        return childA, childB

    def reciprocalExchangeMutation(self, ind):
        """
        Your Reciprocal Exchange Mutation implementation
        """
        if random.random() > self.mutationRate:
            return ind

        idx1 = random.randint(0, self.genSize-1)
        idx2 = random.randint(0, self.genSize-1)

        tmp = ind.genes[idx1]
        ind.genes[idx1] = ind.genes[idx2]
        ind.genes[idx2] = tmp
        
        return ind

    def inversionMutation(self, ind):
        """
        Your Inversion Mutation implementation
        """
        if random.random() > self.mutationRate:
            return ind

        idx1 = random.randint(0, self.genSize-1)
        idx2 = random.randint(0, self.genSize-1)

        if idx1 < idx2:
            ind.genes[idx1:idx2+1] = ind.genes[idx1:idx2+1][::-1]
        else:
            ind.genes[idx2:idx1+1] = ind.genes[idx2:idx1+1][::-1]
        return ind

    def crossover(self, indA, indB):
        """
        Executes a 1 order crossover and returns a new individual
        """
        child = []
        tmp = {}

        indexA = random.randint(0, self.genSize-1)
        indexB = random.randint(0, self.genSize-1)

        for i in range(0, self.genSize):
            if i >= min(indexA, indexB) and i <= max(indexA, indexB):
                tmp[indA.genes[i]] = False
            else:
                tmp[indA.genes[i]] = True
        aux = []
        for i in range(0, self.genSize):
            if not tmp[indB.genes[i]]:
                child.append(indB.genes[i])
            else:
                aux.append(indB.genes[i])
        child += aux
        return child

    def updateMatingPool(self):
        """
        Updating the mating pool before creating a new generation
        """
        self.matingPool = []
        for ind_i in self.population:
                self.matingPool.append( ind_i.copy() )
    
    def getElite(self):
        eliteList = []
        eliteList = sorted(self.population, key=lambda x: x.getFitness(), reverse=True)[:self.elite]
        return eliteList

    def newGeneration(self):
        """
        Creating a new generation
        1. Selection
        2. Crossover
        3. Mutation
        """

        eliteList = []
        if self.elite:
            eliteList = self.getElite()
        
        nextPopulation = []
            
        while (len(nextPopulation)) < self.popSize:
            if self.configuration in [1, 2]:
                indA, indB = self.randomSelection()
            elif self.configuration in [3, 4, 5, 6]:
                indA, indB = self.stochasticUniversalSampling()

            #
            # Calls for crossover
            #
            if self.configuration in [1, 3, 6]:
                c1, c2 = self.uniformCrossover(indA, indB)
            elif self.configuration in [2, 4, 5]:
                c1, c2 = self.pmxCrossover(indA, indB)

            #
            # Create child Individuals
            #
            if self.offsprings == 1:
                child1 = Individual(self.genSize, _genes=c1)
                offspringList = [child1]
            else:
                child1 = Individual(self.genSize, _genes=c1)
                child2 = Individual(self.genSize, _genes=c2)
                offspringList = [child1, child2]


            for child in offspringList:
                #
                # Check if best solution is found before mutation
                #
                if self.updateBest(child):
                    nextPopulation.append(child.copy())

                #
                # Calls for mutation
                #
                if self.configuration in [2, 3, 4]:
                    child = self.reciprocalExchangeMutation(child)
                elif self.configuration in [1, 5, 6]:
                    child = self.inversionMutation(child)
                
                #
                # Replace a parent with child
                #
                nextPopulation.append(child)

                child.computeFitness()
                self.updateBest(child)
        
        #
        # Let population be eliteList + nextPopulation
        #
        self.population = []

        if self.elite:
            self.population = eliteList + nextPopulation
            self.population = sorted(self.population, key=lambda x: x.getFitness())[self.elite:]
        else:
            self.population = nextPopulation

    def GAStep(self):
        self.updateMatingPool()
        self.newGeneration()

    def search(self):
        print('{} - Search started'.format(datetime.now()))

        while self.iteration <= self.maxIterations:
            self.iteration += 1
            print('{} - iteration {} started'.format(datetime.now(), self.iteration))
            self.GAStep()

        print ("\nTotal iterations: ",self.iteration)
        print ("Best Solution: ", self.best.getCost())

        print('{} - Search completed'.format(datetime.now()))


parser = argparse.ArgumentParser()

parser.add_argument('-n', action='store', dest='n', type=int, help='Board size')
parser.add_argument('-c', action='store', dest='configuration', type=int, choices=range(1, 7), help='GA configuration')
parser.add_argument('-p', action='store', dest='popSize', type=int, help='Population size')
parser.add_argument('-m', action='store', dest='mutationRate', type=float, help='Mutation rate')
parser.add_argument('-i', action='store', dest='maxIterations', type=int, help='Maximum iterations per run')
parser.add_argument('-o', action='store', dest='offsprings', type=int, choices=[1, 2], help='Offsprings per crossover (1 or 2)')
parser.add_argument('-e', action='store', dest='elite', type=int, help='Number of elite')
parser.add_argument('-r', action='store', dest='runs', type=int, help='Number of runs')

args = parser.parse_args()

for _ in range(args.runs):
    ga = GA(args.n, args.configuration, args.popSize, args.mutationRate, args.maxIterations, args.offsprings, args.elite)
    ga.search()
