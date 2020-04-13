README
------

Pre-requisites:
---------------

Assigment was developed using Python 3.7
Heuristic Insertion method requries numpy package

    import random
    from Individual import *
    import sys
    import numpy as np

    from datetime import datetime
    import statistics


Execution:
----------
python TSP_R00183658.py [instance] [configuration] [popSize] [mutationRate] [maxIterations] [offsprings] [elite] [runs]

In order to run configuration 1, execute the program as follows:
python TSP_R00183658.py inst-4.tsp 1 100 0.1 500 1 0 5


Output:
-------
The program produces multiple output files:
  1. On completion a solution file is written for the best solution found (cost and sequence of cities)
     e.g.: 20191020-184214-inst-4.tsp-1-100-0.1-500-1-0-solution.tsp
  2. For each run a file is written with statistics per iteration, e.g. timestamp, best fitness, average fitness
     e.g.: 20191020-184214-inst-4.tsp-1-100-0.1-500-1-0.csv


Files:
------

TSP_R00183658.py
  - Implements the GA search including core assignment tasks for crossover, mutation and selection
  - Heuristing insertion requires numpy!

Individual.py
  - Represents a candidate solution class for GA search.
  - Modified template to:
    - Use objective and fitness function
    - Avoid gene shuffling when genes / chromosomes are provided
    - getCost() method

inst-4.tsp, inst-6.tsp, inst-16.tsp 
  - TSP instances relevant for my assignment

inst-test.tsp 
  - Sample tsp file used for debugging crossover and mutation operations

SAMPLE.txt
  - Includes program runs used for the GA analysis
