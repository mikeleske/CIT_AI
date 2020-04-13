"""
Author: Mike Leske
file: CNF.py

Verify if a proposed solution satisfies a CNF problem.
"""

class CNF():
    def __init__(self, cnf_file, sol_file):
        self.cnf_file = cnf_file
        self.sol_file = sol_file
        self.num_vars = 0
        self.num_clauses = 0
        self.clauses = []
        self.solution = {}

        self.read_cnf(cnf_file)
        self.read_sol(sol_file)
    
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
        
        self.clauses = self.clauses[:-1]

    def read_sol(self, sol_file):
        with open(sol_file) as f:
            sol = ''
            for line in f.readlines():
                if line.startswith('v'):
                    sol += line.strip().replace('v ', '')
                    sol = sol.split(' 0')[0].strip()
            
            self.solution = [ literal for literal in sol.split() ]

    def eval_solution(self):
        if len(self.solution) != self.num_vars:
            exit('Number of problem variables does not match solution variables.')
        
        unsat_clauses = 0
        for clause in self.clauses:
            if [ literal in self.solution for literal in clause ].count(True) == 0:
                unsat_clauses += 1
        return unsat_clauses
