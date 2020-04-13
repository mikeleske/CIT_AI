"""
Author: Mike Leske
file: CNF.py

CNF class representing the current SAT state, evaluate new solutions 
and generate neighbors by flipping bits.
"""

import random
from itertools import count

class CNF():
    def __init__(self, cnf_file, sol_file):
        self.cnf_file = cnf_file
        self.num_vars = 0
        self.vars = []
        self.num_clauses = 0
        self.clauses = []
        self.n = count()

        self.best_solution = {}

        self.read_cnf(cnf_file)
    
    def read_cnf(self, cnf_file):
        with open(cnf_file) as f:
            cnf = ''
            for line in f.readlines():
                if line.startswith('c'): continue
                elif line.startswith('%'): break
                elif line.startswith('p'):
                    self.num_vars = int(line.split()[2])
                    self.num_clauses = int(line.split()[3])
                else:
                    cnf += line.strip()
            
            for clause in cnf.split(' 0'):
                self.clauses.append([ literal for literal in clause.split() ])
            
            self.vars = sorted(set([ abs(int(i)) for i in cnf.replace(' 0', ' ').split() ]))
        
        self.clauses = self.clauses[:-1]
    
    def initial_solution(self):
        self.best_solution['assignment'] = []
        self.best_solution['unsat'] = 0

        for var in self.vars:
            #self.best_solution['assignment'][var] = bool(random.getrandbits(1))
            if round(random.uniform(0, 1)):
                self.best_solution['assignment'].append(var)
            else:
                self.best_solution['assignment'].append(var * -1)
        
        self.best_solution['unsat'] = self.eval_solution(self.best_solution['assignment'])
        #print('Initial unsatisfied clauses:', self.get_unsat())

    def get_unsat(self):
        return self.best_solution['unsat']

    def eval_solution(self, solution):
        #if len(self.solution) != self.num_vars:
        #    exit('Number of problem variables does not match solution variables.')
        
        unsat_clauses = 0
        for clause in self.clauses:
            if [ int(literal) in solution for literal in clause ].count(True) == 0:
                unsat_clauses += 1
                if self.get_unsat() > 0 and self.get_unsat() < unsat_clauses:
                    return unsat_clauses
        return unsat_clauses

    def next_neighbor(self):
        flip_var = next(self.n)
        if flip_var <= len(self.best_solution['assignment']) - 1:
            candidate = self.best_solution['assignment'].copy()
            candidate[flip_var] = candidate[flip_var] * -1
            return candidate
    
    def set_solution(self, assignment, unsat):
        self.best_solution['assignment'] = assignment
        self.best_solution['unsat'] = unsat
    
    def reset_counter(self):
        self.n = count()
