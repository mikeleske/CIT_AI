"""
Author: Mike Leske
file: Leske_183658_WalkSAT.py

Kicker for LocalSearch based on WalkSAT
"""

from LocalSearch import LocalSearch
import sys
import time
import numpy as np
import random

from utils import write_statistics, plot_stats, plot_unsat, plot_rtd

myStudentNum = 183658 
random.seed(myStudentNum)


if len(sys.argv) < 7:
    print ("Error - Incorrect input")
    print ("Expecting python Leske_183658_WalkSAT.py [cnf_file] [executions] [max_iterations] [restarts] [p] [tl]")
    sys.exit(0)


cnf_file        = sys.argv[1]
executions      = int(sys.argv[2])
max_iterations  = int(sys.argv[3])
restarts        = int(sys.argv[4])
p               = float(sys.argv[5])
tl              = int(sys.argv[6])
alg             = 'walksat'

DEBUG           = False

exec_stats = []
exec_sum = 0

search = LocalSearch(cnf_file, restarts, max_iterations, alg, p=p, tl=tl, DEBUG=DEBUG)

for execution in range(executions):
    # Execute one search
    t_execution_start = time.perf_counter_ns()
    restart, iteration, solution = search.run()
    d = int((time.perf_counter_ns() - t_execution_start) / 1000)
    
    # Add successful search to stats
    if solution:
        exec_stats.append((execution, restart, iteration, restart*max_iterations + iteration, d, solution))

    # Sum up timedeltas
    exec_sum += d


write_statistics(alg, executions, exec_stats, exec_sum)

#
# Uncomment if any of the below plot should be generated.
#
#plot_stats(exec_stats, exec_sum, cnf_file, executions, restarts, max_iterations, alg, p=p)
#plot_unsat(search.search_unsat)
#plot_rtd(exec_stats, exec_sum, cnf_file, executions, restarts, max_iterations, alg, p=p)
