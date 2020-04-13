"""
Author: Mike Leske
file: CNF.py

CNF class representing the current SAT state.
Simulating flip operations. Updating SAT solution state based on LS algo decision.
"""

import random
from itertools import count
from collections import defaultdict
import pprint as pp

class CNF():
    def __init__(self, cnf_file, sol_file):
        self.cnf_file = cnf_file
        self.num_variables = 0
        self.variables = []
        self.num_clauses = 0
        self.clauses = defaultdict()
        self.clauseDict = defaultdict()

        self.solution = {}

        self.read_cnf(cnf_file)

    def read_cnf(self, cnf_file):
        with open(cnf_file) as f:
            cnf = ''
            for line in f.readlines():
                if line.startswith('c'): continue
                elif line.startswith('%'): break
                elif line.startswith('p'):
                    self.num_variables = int(line.split()[2])
                    self.num_clauses = int(line.split()[3])
                else:
                    cnf += line.strip()
            
            #
            # Assumption that every variable appears in problem instance.
            # Preprocessing can reduce this list.
            #
            self.variables = [ variable for variable in range(1, self.num_variables + 1) ]
            
            #
            # Initialize for each possible variable (variable or -variable) a list.
            #
            for variable in self.variables:
                self.clauseDict[int(variable)] = []
                self.clauseDict[int(-variable)] = []
            
            #
            # For each variable of a clause, append the clause to the appropriate 
            # variable list in clause dictionary. 
            #
            for clause in cnf.split(' 0')[:-1]:
                cl = tuple(clause.split())
                self.clauses[cl] = False
                for var in cl:
                    self.clauseDict[int(var)].append(cl)

        
    def initial_solution(self):
        self.solution['assignment'] = []
        self.solution['unsat'] = 0

        #
        # For each variable toss a coin an set the variable assignment to True or False
        #
        for var in self.variables:
            if round(random.uniform(0, 1)):
                self.solution['assignment'].append(var)
            else:
                self.solution['assignment'].append(var * -1)

        #
        # Per clause count the number of variables in solution assignment.
        # Bool function maps whole clause to True or False. 
        #
        for clause, _ in self.clauses.items():
            self.clauses[clause] = bool(sum([ int(var) in self.solution['assignment'] for var in clause ]))

        #
        # Count unsatisfied clauses
        #
        self.solution['unsat'] = self.num_clauses - sum([ v for v in self.clauses.values() ])
        
    def get_unsat_count(self):
        return self.solution['unsat']
    
    def get_unsat_clauses(self):
        return [k for k, v in self.clauses.items() if v is False]

    def flip_sim(self, var):
        #
        # Create candidate assignment, flip chosen bit.
        #
        idx = abs(var) - 1
        candidate = self.solution['assignment'].copy()
        candidate[idx] *= -1
        
        candidate_clauses = self.clauses.copy()

        pos_gain = 0
        neg_gain = 0
        
        #
        # Get clauses that include the flipped variable, set state True, increase pos_gain
        #
        for clause in self.clauseDict[candidate[idx]]:
            candidate_clauses[clause] = True
            if self.clauses[clause] == False: pos_gain += 1

        #
        # Get clauses that include the inverse flipped variable
        # Check not True set state to False
        # Increment neg_gain if clause was positive before.
        #       
        for clause in self.clauseDict[-candidate[idx]]:
            if not sum([ int(var) in candidate for var in clause ]):
                candidate_clauses[clause] = False
                if self.clauses[clause] == True: neg_gain += 1

        return pos_gain, neg_gain, candidate_clauses
    
    def update_solution(self, flip_var, update):
        self.solution['assignment'][flip_var - 1] = self.solution['assignment'][flip_var - 1] * -1
        self.solution['unsat'] -= update['net_gain']
        self.clauses = update['candidate_clauses']

    def validate_solution(self):
        for clause, _ in self.clauses.items():
            if not sum([ int(var) in self.solution['assignment'] for var in clause ]):
                return False
        return True