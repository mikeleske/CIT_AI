"""
Author: Mike Leske
file: SatChecker.py

Verify if a proposed solution satisfies a CNF problem.
"""

from CNF import CNF
import sys

cnf_file = sys.argv[1]
sol_file = sys.argv[2]

cnf = CNF(cnf_file, sol_file)
res = cnf.eval_solution()

print('Total       clauses:', len(cnf.clauses))
print('Unsatisfied clauses:', res)