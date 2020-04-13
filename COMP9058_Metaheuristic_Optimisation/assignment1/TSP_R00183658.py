"""
Author: Mike Leske - R00183658
file: 
Rename this file to TSP_x.py where x is your student number 
"""

import random
from Individual import *
import sys
import numpy as np

from datetime import datetime
import statistics

myStudentNum = 183658 # Replace 12345 with your student number
random.seed(myStudentNum)

_BEST = None
_BESTLIST = []

class BasicTSP:
    def __init__(self, _fName, _configuration, _popSize, _mutationRate, _maxIterations, _offsprings, _elite):
        """
        Parameters and general variables
        """

        self.population     = []
        self.matingPool     = []
        self.best           = None
        self.popSize        = _popSize
        self.genSize        = None
        self.mutationRate   = _mutationRate
        self.maxIterations  = _maxIterations
        self.iteration      = 0
        self.fName          = _fName
        self.data           = {}

        self.configuration  = _configuration
        self.offsprings     = _offsprings
        self.elite          = _elite

        self.tracker = {}

        self.readInstance()

        if self.configuration in [1, 2, 3, 4, 5, 6]:
            self.initPopulation()
        elif self.configuration in [7, 8]:
            print('{} - Start heuristic population'.format(datetime.now()))
            self.heuristicPopulation()
            print('{} - Completed heuristic population'.format(datetime.now()))

    def readInstance(self):
        """
        Reading an instance from fName
        """
        file = open(self.fName, 'r')
        self.genSize = int(file.readline())
        self.data = {}
        for line in file:
            (id, x, y) = line.split()
            self.data[int(id)] = (int(x), int(y))
        file.close()


    def initPopulation(self):
        """
        Creating random individuals in the population
        """
        for _ in range(0, self.popSize):
            individual = Individual(self.genSize, self.data)
            individual.computeFitness()
            self.population.append(individual)
            self.updateBest(individual)

        print ("Best initial sol: ",self.best.getFitness())

    def heuristicPopulation(self):
        def load_data(f):
            count = np.loadtxt(f, delimiter=' ', max_rows=1)
            cities = np.loadtxt(f, delimiter=' ', dtype={'names': ('id', 'x', 'y'), 'formats': ('int', 'float', 'float')}, skiprows=1)
            return count, cities

        count, cities = load_data(self.fName)

        for _ in range(0, self.popSize):
            idx = int(random.randint(1, count))
            not_visited = cities[cities['id'] != idx]['id'].tolist()
            tsp_list = [idx]
            
            while not_visited:
                cur_x = cities[cities['id'] == idx]['x']
                cur_y = cities[cities['id'] == idx]['y']
                remaining_cities = cities[np.isin(cities['id'], not_visited)]
                distance = np.array([
                    remaining_cities['id'],
                    np.around(np.sqrt(np.power(remaining_cities['x'] - cur_x, 2) + np.power(remaining_cities['y'] - cur_y, 2)))
                ])
                tsp_list.append( distance[0][distance[1].argmin()].astype(int) )
                not_visited.remove(distance[0][distance[1].argmin()])
                idx = int(distance[0][distance[1].argmin()])

            individual = Individual(self.genSize, self.data, _genes=tsp_list)
            individual.computeFitness()
            self.population.append(individual)
            self.updateBest(individual)

        print ("Best initial sol : ",self.best.getFitness())
        print ("Best initial cost: ",self.best.getCost())

    def updateBest(self, candidate):
        if self.best == None or candidate.getFitness() > self.best.getFitness():
            self.best = candidate.copy()
            print ("  >> iteration: ", self.iteration, "best: ", self.best.getFitness(), self.best.getCost())

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
        #sortedPop = sorted(self.population, key=lambda x: x.getFitness(), reverse=True)

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
        #mapA = []
        #mapB = []

        idx1, idx2 = sorted(random.sample(range(0, self.genSize), 2))
        mapA = childA[idx1:idx2+1] = indB.genes[idx1:idx2+1].copy()
        mapB = childB[idx1:idx2+1] = indA.genes[idx1:idx2+1].copy()
        #mapA = indB.genes[idx1:idx2+1].copy()  ## Speedup when mapX and childX are copied together from indX
        #mapB = indA.genes[idx1:idx2+1].copy()  ## Speedup when mapX and childX are copied together from indX

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

    def mutation(self, ind):
        """
        Mutate an individual by swaping two cities with certain probability (i.e., mutation rate)
        """
        if random.random() > self.mutationRate:
            return
        indexA = random.randint(0, self.genSize-1)
        indexB = random.randint(0, self.genSize-1)

        tmp = ind.genes[indexA]
        ind.genes[indexA] = ind.genes[indexB]
        ind.genes[indexB] = tmp

        ind.computeFitness()
        self.updateBest(ind)

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
            
        #while (len(eliteList) + len(nextPopulation)) < self.popSize:
        while (len(nextPopulation)) < self.popSize:
            if self.configuration in [1, 2]:
                indA, indB = self.randomSelection()
            elif self.configuration in [3, 4, 5, 6, 7, 8]:
                indA, indB = self.stochasticUniversalSampling()

            #
            # Calls for crossover
            #
            if self.configuration in [1, 3, 6, 8]:
                c1, c2 = self.uniformCrossover(indA, indB)
            elif self.configuration in [2, 4, 5, 7]:
                c1, c2 = self.pmxCrossover(indA, indB)

            #
            # Create child Individuals
            #
            if self.offsprings == 1:
                child1 = Individual(self.genSize, self.data, _genes=c1)
                offspringList = [child1]
            else:
                child1 = Individual(self.genSize, self.data, _genes=c1)
                child2 = Individual(self.genSize, self.data, _genes=c2)
                offspringList = [child1, child2]

            for child in offspringList:
                #
                # Calls for mutation
                #
                if self.configuration in [2, 3, 4, 7]:
                    child = self.reciprocalExchangeMutation(child)
                elif self.configuration in [1, 5, 6, 8]:
                    child = self.inversionMutation(child)
                
                #
                # Replace a parent with child
                #
                nextPopulation.append(child)

                child.computeFitness()
                self.updateBest(child)
                
                #
                # Validation code to produce valid offsprings
                #
                if np.unique(child.genes).shape[0] != self.genSize:
                    print('oops:', np.unique(child.genes).shape[0])
        
        #
        # Let population be eliteList + nextPopulation
        #
        self.population = []

        if self.elite:
            self.population = eliteList + nextPopulation
            self.population = sorted(self.population, key=lambda x: x.getFitness())[self.elite:]
        else:
            self.population = nextPopulation
       

    def update_tracker(self):
        self.tracker[self.iteration] = (datetime.now(), [ ind.getCost() for ind in self.population ])

    def write_tracker(self):
        filename = '{}-{}-{}-{}-{}-{}-{}-{}.csv'.format(datetime.now().strftime("%Y%m%d-%H%M%S"), self.fName, self.configuration, self.popSize, 
                                                    self.mutationRate, self.maxIterations, self.offsprings, self.elite)
        f = open(filename, "w+")
        f.write('{}\n'.format(str(self.best.getCost())))
        for k, v in self.tracker.items():
            f.write('{},{},{},{},{}\n'.format(str(v[0]), k, min(v[1]), statistics.mean(v[1]), statistics.mean(v[1][:10])))
        f.write(str(self.best.genes))
        f.close()

    def write_solution(self):
        filename = '{}-{}-{}-{}-{}-{}-{}-{}-solution.tsp'.format(datetime.now().strftime("%Y%m%d-%H%M%S"), self.fName, self.configuration, self.popSize, 
                                                    self.mutationRate, self.maxIterations, self.offsprings, self.elite)
        f = open(filename, "w+")
        f.write('{}\n'.format(str(round(_BEST.getCost()))))
        for gene in _BEST.genes: f.write(str(gene)+'\n')
        f.close()

    def GAStep(self):
        self.updateMatingPool()
        self.newGeneration()

    def search(self):
        #print('{} - Search started'.format(datetime.now()))

        self.iteration = 0
        while self.iteration < self.maxIterations:
            #print('{} - iteration {} started'.format(datetime.now(), self.iteration))
            self.GAStep()

            #print ("\nBest so far =============")
            #print ("Iteration: "+str(self.iteration))
            #print ("Fitness: "+str(self.best.getFitness()))
            #print ("Cost: "+str(self.best.getCost()))
            #print ("=========================\n")

            self.update_tracker()
            self.iteration += 1

        print ("\nTotal iterations: ",self.iteration)
        print ("Best Solution: ", self.best.getCost())

        #print('{} - Search completed'.format(datetime.now()))

        self.write_tracker()

if len(sys.argv) < 8:
    print ("Error - Incorrect input")
    print ("Expecting python TSP_R00183658.py [instance] [configuration] [popSize] [mutationRate] [maxIterations] [offsprings] [elite] [runs]")
    sys.exit(0)

problem_file    = sys.argv[1]
configuration   = int(sys.argv[2])
popSize         = int(sys.argv[3])
mutationRate    = float(sys.argv[4])
maxIterations   = int(sys.argv[5])
offsprings      = int(sys.argv[6])
elite           = int(sys.argv[7])
runs            = int(sys.argv[8])

if not(0 < configuration <= 9):
    print ("Error - Incorrect input")
    print ("The experiment configuration must be in the range 1..8.")
    sys.exit(0)

for _ in range(runs):
    ga = BasicTSP(problem_file, configuration, popSize, mutationRate, maxIterations, offsprings, elite)
    ga.search()

    if _BEST == None or ga.best.getFitness() > _BEST.getFitness():
        _BEST = ga.best

    _BESTLIST.append(ga.best.getCost())

ga.write_solution()


# Print summary stats:
print('\nBest solution:', round(_BEST.getCost()))
print('Avg  solution:', round(sum(_BESTLIST)/len(_BESTLIST)))