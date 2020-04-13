"""
Author: Mike Leske
file: LocalSearch.py

Driver for local search. 
"""

from CNF import CNF
from collections import defaultdict
from itertools import count
import numpy as np
import sys, random

cnf_file = sys.argv[1]
sol_file = ''


## Local Search
max_tries = 100
max_flips = 50

cnf = CNF(cnf_file, sol_file)

total = count()

for restart in range(0, max_tries):
    #print('\nTry:', restart)

    #
    # Create an initial solution and check if it satisfies the formula
    #
    cnf.initial_solution()
    next(total)
    if cnf.best_solution['unsat'] == 0:
        print('\nOptimal initial solution:', total, cnf.best_solution['assignment'])
        exit()

    for flip in range(0, max_flips):
        cnf.reset_counter()

        candidates = []
        candidates_unsat = []

        #
        # next_neighbor() returns 1 neighbor after the after so that we can check
        # if it satisfies the formula early. Skip full neighborhood generation in
        # problems with very large neighborhoods.
        #
        candidate = cnf.next_neighbor()
        next(total)
        while candidate:
            candidate_unsat = cnf.eval_solution(candidate)
            
            if candidate_unsat == 0:
                cnf.set_solution(candidate, candidate_unsat)
                print('\nOptimal solution:', total, cnf.best_solution['assignment'])
                exit()
            else:
                candidates.append(candidate)
                candidates_unsat.append(candidate_unsat)

            candidate = cnf.next_neighbor()

        #
        # Find solution(s) with lowest unsat value.
        # Set random solution with lowest unsat as new best solution.
        #
        if cnf.best_solution['unsat'] > min(candidates_unsat):
            idx = [i for i, v in enumerate(candidates_unsat) if v == min(candidates_unsat)]
            cnf.set_solution(candidates[random.choice(idx)], min(candidates_unsat))
        else:
            break

#print('Total       clauses:', len(cnf.clauses))
#print('Unsatisfied clauses:', res)