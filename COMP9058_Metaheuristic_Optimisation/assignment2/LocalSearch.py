"""
Author: Mike Leske
file: LocalSearch.py

Driver for local search. 
"""

from CNF import CNF
from collections import defaultdict
import sys, random
import numpy as np
import time

class LocalSearch():
    def __init__(self, cnf_file, restarts, max_iterations, alg, wp=0, p=0, tl=0, DEBUG=False):
        self.cnf_file = cnf_file
        self.sol_file = ''

        self.restarts = restarts
        self.max_iterations = max_iterations
        self.wp = wp
        self.p = p
        self.tl = tl

        self.alg = alg
        self.DEBUG = DEBUG

        self.cnf = CNF(self.cnf_file, self.sol_file)
        self.search_unsat = []

    def gwsat(self, cnf):
        #
        # updateDict stores simulation data to make search step decisions
        #
        updateDict = defaultdict()
        updateDict['candidate_vars'] = []
        updateDict['gains'] = []

        #
        # With P = wp we flip a random variable from unsat clauses.
        # updateDict['candidate_vars'] is simply reduced to this random uniform choice.
        # Else updateDict['candidate_vars'] is the set of all variable.
        #
        # From Assignment doc:
        #   The random walk component is that for each step of the algorithm, the variable to be flipped is
        #   selected randomly (from all variables involved in at least one unsatisfied clause) with
        #   probability wp, ...
        #
        # Note: 
        #   This is in slight contradiction to Lecture slides (L9 page 38), but net net in both cases 
        #   a random variable from an unsat clause is selected
        # 
        if random.uniform(0, 1) < self.wp:
            #clause = random.choice(cnf.get_unsat_clauses())
            #updateDict['candidate_vars'] = [int(random.choice(clause))]
            updateDict['candidate_vars'] = list(set(abs(int(x)) for l in cnf.get_unsat_clauses() for x in l))
            updateDict['candidate_vars'] = [random.choice(updateDict['candidate_vars'])]
        else:
            updateDict['candidate_vars'] = self.cnf.variables

        #
        # Loop over all candidate_vars and simulate: pos_gain, neg_gain, candidate_clauses
        # 
        b0 = cnf.get_unsat_count()
        for var in updateDict['candidate_vars']:            
            pos_gain, neg_gain, candidate_clauses = cnf.flip_sim(var)
            b1 = b0 - pos_gain + neg_gain
            net_gain = b0 - b1
            
            # Do housekeeping for later selection of best variable to flip
            updateDict['gains'].append(net_gain)
            updateDict[var] = defaultdict()
            updateDict[var]['net_gain'] = net_gain
            updateDict[var]['candidate_clauses'] = candidate_clauses
        
        #
        # From all simulations choose the the one with the best net_gain
        # In tie situation, uniformly choose one of best vars
        #
        best_idx = [i for i, v in enumerate(updateDict['gains']) if v == max(updateDict['gains'])]
        flip_var = updateDict['candidate_vars'][random.choice(best_idx)]

        if self.DEBUG:
            self.gwsat_debug_var_selection(flip_var, updateDict)

        return flip_var, updateDict

    def gwsat_debug_var_selection(self, flip_var, updateDict):
        print('\ntotal_unsat:   ', self.cnf.get_unsat_count())
        print('candidate_vars:', updateDict['candidate_vars'])
        print('gains:         ', updateDict['gains'])
        print('flip_var:      ', flip_var)

    def walksat_skc(self, cnf, iteration, tabu):
        #
        # 1. Select an unsatisfied clause BC (uniformly at random)
        #
        clause = random.choice(cnf.get_unsat_clauses())

        #
        # updateDict stores simulation data to make search step decisions
        # It stores b0, b1, net_gain and the future clauses state, so that
        # flip can be made without further calculations.
        #
        updateDict = defaultdict()
        updateDict['candidate_vars'] = list(set(abs(int(l)) for l in clause))
        updateDict['gains'] = []

        #
        # Filter out candidates blocked by tabu state
        # A var is not tabu'ed, if its tabu list entry in not None (was not flipped before)
        # and if flip id is larger than tabu counter for variable.
        #
        if self.DEBUG:
            print('\nIteration                 :', iteration)
            print('Tabu List                 :', tabu)
            print('candidate_vars before tabu:', updateDict['candidate_vars'])

        updateDict['candidate_vars'][:] = [
            var for var in updateDict['candidate_vars'] 
            if not(tabu[var - 1] is not None and (iteration <= tabu[var - 1])) 
        ]

        if self.DEBUG:
            print('candidate_vars after  tabu:', updateDict['candidate_vars'])

        #
        # No flip possible, because all variables are tabued
        #
        if len(updateDict['candidate_vars']) == 0:
            return None, None

        #
        # Loop over all non-tabu candidate_vars and simulate: bo, b1, candidate_clauses
        # 
        b0 = cnf.get_unsat_count()
        neg_gain_list = []
        for var in updateDict['candidate_vars']:
            pos_gain, neg_gain, candidate_clauses = cnf.flip_sim(var)
            b1 = b0 - pos_gain + neg_gain
            net_gain = b0 - b1
            neg_gain_list.append(neg_gain)
            
            # Do housekeeping for later selection of best variable to flip
            updateDict['gains'].append(net_gain)
            updateDict[var] = defaultdict()
            updateDict[var]['net_gain'] = net_gain
            updateDict[var]['candidate_clauses'] = candidate_clauses

        #
        # 2. If at least one variable in BC has negative gain of 0 (i.e. flipping the variable does not
        #    make any clause that is currently satisfied go unsat), randomly select one of these
        #    variables.
        #
        flip_var = ''
        neg_gain_0 = [i for i, v in enumerate(neg_gain_list) if v == 0]
        if neg_gain_0:
            flip_var = updateDict['candidate_vars'][int(random.choice(neg_gain_0))]
        
        #
        # 3. Otherwise, with probability p, select random variable from BC to flip, and with
        #    probability (1-p), select variable in BC with minimal negative gain (break ties randomly)
        #
        else:
            if random.uniform(0, 1) < self.p:
                if self.DEBUG:
                    print('random walk')
                flip_var = random.choice(updateDict['candidate_vars'])
            else:
                best_idx = [i for i, v in enumerate(neg_gain_list) if v == min(neg_gain_list)]
                flip_var = updateDict['candidate_vars'][random.choice(best_idx)]

        if self.DEBUG:
            print('candidate_vars neg_gain   :', neg_gain_list)
            print('selected variable         :', flip_var)

        tabu[flip_var - 1] = iteration + self.tl
        return flip_var, updateDict

    def run(self):
        for restart in range(0, self.restarts):
            #
            # Create an initial solution and check if it satisfies the formula
            #
            self.cnf.initial_solution()

            if self.cnf.solution['unsat'] == 0:
                return (restart, 0, self.cnf.solution['assignment'])

            #
            # Create tabu list
            # 
            if self.alg == 'walksat':
                tabu = [ None for x in self.cnf.variables ]

            #
            # Iterate max_iterations times
            # 
            #
            for iteration in range(1, self.max_iterations + 1):
                if self.alg == 'gwsat': 
                    flip_var, updateDict = self.gwsat(self.cnf)
                elif self.alg == 'walksat':
                    flip_var, updateDict = self.walksat_skc(self.cnf, iteration, tabu)
                else:
                    return (restart, iteration, 'Invalid algorithm provided.')

                #
                # Update the CNF object
                #
                if flip_var:
                    self.cnf.update_solution(abs(flip_var), updateDict[flip_var])

                #
                # Track number of unsat_clauses.
                # Only included to get stats for report. Can be safely commented out.
                #
                self.search_unsat.append(self.cnf.solution['unsat'])

                #
                # Perform final sanity check when unsat count equals 0
                #
                if self.cnf.get_unsat_count() == 0:
                    if self.cnf.validate_solution():
                        return (restart, iteration, self.cnf.solution['assignment'])
                    else:
                        return (restart, iteration, 'Solution failed verification.')

        return (self.restarts, self.max_iterations, None)