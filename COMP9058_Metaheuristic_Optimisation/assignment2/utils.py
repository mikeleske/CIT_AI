"""
Author: Mike Leske
file: utils.py

Print stats, plot diagrams
"""

import numpy as np 
import matplotlib.pyplot as plt
import statistics as s

def write_statistics(alg, executions, exec_stats, exec_sum):
    restarts  = [ e[1] for e in exec_stats ]
    max_iter  = [ e[2] for e in exec_stats ]
    tot_iter  = [ e[3] for e in exec_stats ]
    runtime   = [ e[4] for e in exec_stats ]

    stats = []
    stats.append('Total : {} µsec\n'.format(exec_sum))
    stats.append('Mean  : {} µsec\n'.format(int(exec_sum/executions)))
    stats.append('Success rate: {:.2%}\n'.format(len(tot_iter)/executions))
    stats.append('\n')
    stats.append('Total iter: {}\n'.format(sum(tot_iter)))
    stats.append('Avg iter time: {} µsec\n'.format( round(sum(runtime)/sum(tot_iter)) ))
    stats.append('\n')
    if executions > 1:
        stats.append('Restarts:\n')
        stats.append('  Sum   : {}\n'.format(sum(restarts)))
        stats.append('  Min   : {}\n'.format(min(restarts)))
        stats.append('  Max   : {}\n'.format(max(restarts)))
        stats.append('  Mean  : {}\n'.format(round(s.mean(restarts), 1)))
        stats.append('  Median: {}\n'.format(s.median(restarts)))
        stats.append('  STDEV : {}\n'.format(round(s.stdev(restarts), 1)))
        stats.append('\nTotal iterations:\n')
        stats.append('  Sum   : {}\n'.format(sum(tot_iter)))
        stats.append('  Min   : {}\n'.format(min(tot_iter)))
        stats.append('  Max   : {}\n'.format(max(tot_iter)))
        stats.append('  Mean  : {}\n'.format(round(s.mean(tot_iter), 1)))
        stats.append('  Median: {}\n'.format(s.median(tot_iter)))
        stats.append('  STDEV : {}\n'.format(round(s.stdev(tot_iter), 1)))
        stats.append('  VarCo : {}\n'.format(round(s.stdev(tot_iter) / s.mean(tot_iter), 3)))
        stats.append('  Q0.1  : {}\n'.format(int(np.quantile(tot_iter, .1))))
        stats.append('  Q0.25 : {}\n'.format(int(np.quantile(tot_iter, .25))))
        stats.append('  Q0.5  : {}\n'.format(int(np.quantile(tot_iter, .5))))
        stats.append('  Q0.75 : {}\n'.format(int(np.quantile(tot_iter, .75))))
        stats.append('  Q0.9  : {}\n'.format(int(np.quantile(tot_iter, .9))))
        stats.append('\nIterations to solve:\n')
        stats.append('  Min   : {}\n'.format(min(max_iter)))
        stats.append('  Max   : {}\n'.format(max(max_iter)))
        stats.append('  Mean  : {}\n'.format(round(s.mean(max_iter), 1)))
        stats.append('  Median: {}\n'.format(s.median(max_iter)))
        stats.append('  STDEV : {}\n'.format(round(s.stdev(max_iter), 1)))
        stats.append('  VarCo : {}\n'.format(round(s.stdev(max_iter) / s.mean(max_iter), 3)))
        stats.append('  Q0.1  : {}\n'.format(int(np.quantile(max_iter, .1))))
        stats.append('  Q0.25 : {}\n'.format(int(np.quantile(max_iter, .25))))
        stats.append('  Q0.5  : {}\n'.format(int(np.quantile(max_iter, .5))))
        stats.append('  Q0.75 : {}\n'.format(int(np.quantile(max_iter, .75))))
        stats.append('  Q0.9  : {}\n'.format(int(np.quantile(max_iter, .9))))
        stats.append('\nRuntime in µsec:\n')
        stats.append('  Sum   : {}\n'.format(sum(runtime)))
        stats.append('  Min   : {}\n'.format(min(runtime)))
        stats.append('  Max   : {}\n'.format(max(runtime)))
        stats.append('  Mean  : {}\n'.format(round(s.mean(runtime), 1)))
        stats.append('  Median: {}\n'.format(s.median(runtime)))
        stats.append('  STDEV : {}\n'.format(round(s.stdev(runtime), 1)))
        stats.append('  VarCo : {}\n'.format(round(s.stdev(runtime) / s.mean(runtime), 3)))
        stats.append('  Q0.1  : {}\n'.format(int(np.quantile(runtime, .1))))
        stats.append('  Q0.25 : {}\n'.format(int(np.quantile(runtime, .25))))
        stats.append('  Q0.5  : {}\n'.format(int(np.quantile(runtime, .5))))
        stats.append('  Q0.75 : {}\n'.format(int(np.quantile(runtime, .75))))
        stats.append('  Q0.9  : {}\n'.format(int(np.quantile(runtime, .9))))
    else:
        stats.append('Restarts           : {}\n'.format(restarts[0]))
        stats.append('Total iterations   : {}\n'.format(tot_iter[0]))
        stats.append('Iterations to solve: {}\n'.format(max_iter[0]))

    if alg == 'gwsat':
        with open('stats-gwsat-summary.txt', 'w') as f:
            f.writelines(stats)
        with open('stats-gwsat-detail.txt', 'w') as f:
            f.write("execution; restart; iterations; total_iterations; duration; solution\n")
            for e in exec_stats:
                f.write("%s; %s; %s; %s; %s; %s\n" % (e[0], e[1], e[2], e[3], e[4], e[5]))
    elif alg == 'walksat':
        with open('stats-walksat-summary.txt', 'w') as f:
            f.writelines(stats)
        with open('stats-walksat-detail.txt', 'w') as f:
            f.write("execution; restart; iterations; total_iterations; duration; solution\n")
            for e in exec_stats:
                f.write("%s; %s; %s; %s; %s; %s\n" % (e[0], e[1], e[2], e[3], e[4], e[5]))

def plot_stats(exec_stats, exec_sum, cnf_file, executions, max_restarts, max_iterations, alg, wp=None, p=None, tl=None):
    execution = [ e[0] for e in exec_stats ]
    restarts = [ e[1] for e in exec_stats ]
    max_iter = [ e[2] for e in exec_stats ]
    tot_iter = [ e[3] for e in exec_stats ]

    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(12,8))
    if alg == 'gwsat':
        title = '{}-{}-{}-{}-{}-{}'.format(cnf_file, alg, executions, max_iterations, max_restarts, wp)
    if alg == "walksat":
        title = '{}-{}-{}-{}-{}-{}-{}'.format(cnf_file, alg, executions, max_iterations, max_restarts, p, tl)
    fig.suptitle(title)

    ax1.plot(execution, restarts)
    ax1.grid(b=True, which='both', color='#666666', linestyle='-', alpha=0.1)
    ax1.set_title('Restarts per execution')
    ax1.set_xlabel('Executions')
    ax1.set_ylabel('Restarts')

    ax2.plot(execution, np.cumsum(tot_iter))
    ax2.grid(b=True, which='both', color='#666666', linestyle='-', alpha=0.1)
    ax2.set_title('Cumulative sum of iterations')
    ax2.set_xlabel('Executions')
    ax2.set_ylabel('Iterations')

    ax3.plot(execution, sorted(max_iter))
    ax3.grid(b=True, which='both', color='#666666', linestyle='-', alpha=0.1)
    ax3.set_title('Max iterations to solve')
    ax3.set_xlabel('Executions - Sorted by length')
    ax3.set_ylabel('Iterations')
    
    ax4.plot(execution, sorted(tot_iter))
    ax4.grid(b=True, which='both', color='#666666', linestyle='-', alpha=0.1)
    ax4.set_title('Total iterations per execution')
    ax4.set_xlabel('Executions - Sorted by length')
    ax4.set_ylabel('Iterations')

    ax5.boxplot(max_iter)
    ax5.set_title('Total iterations to solve')
    ax5.set_ylabel('Iterations')

    ax6.boxplot(tot_iter)
    ax6.set_title('Total iterations per execution')
    ax6.set_ylabel('Total iterations per execution')

    plt.tight_layout()
    plt.savefig('./plot-stats.png')
    plt.show()

def plot_unsat(search_unsat):
    plt.plot(search_unsat)
    plt.title('Unsat clauses over time')
    plt.xlabel('Iteration')
    plt.ylabel('Number unsat clauses')
    plt.tight_layout()
    plt.savefig('./plot-unsat.png')
    plt.show()

def plot_rtd(exec_stats, exec_sum, cnf_file, executions, max_restarts, max_iterations, alg, wp=None, p=None, tl=None):
    tot_iter  = [ e[3] for e in exec_stats ]
    runtime   = [ e[4] for e in exec_stats ]

    _, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(12,8))
    
    x_iter = sorted(tot_iter)
    x_run  = sorted(runtime)
    y = np.arange(1, len(tot_iter)+1) / len(tot_iter)

    ax1.plot(x_iter, y, linestyle='dotted')
    ax1.grid(b=True, which='both', color='#666666', linestyle='-', alpha=0.1)
    ax1.set_title('RLD (Search Steps)')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('P(Solve)')

    ax2.plot(x_run, y, linestyle='dotted')
    ax2.grid(b=True, which='both', color='#666666', linestyle='-', alpha=0.1)
    ax2.set_title('RTD (Runtime)')
    ax2.set_xlabel('Runtime in µsec')
    ax2.set_ylabel('P(Solve)')

    ax3.plot(x_iter, y, linestyle='dotted')
    ax3.grid(b=True, which='both', color='#666666', linestyle='-', alpha=0.1)
    ax3.set_title('RLD (Search Steps) Semi Log')
    ax3.set_xlabel('Iterations')
    ax3.set_ylabel('P(Solve)')
    ax3.set_xscale('log')

    ax4.plot(x_run, y, linestyle='dotted')
    ax4.grid(b=True, which='both', color='#666666', linestyle='-', alpha=0.1)
    ax4.set_title('RTD (Runtime) - Semi Log')
    ax4.set_xlabel('Runtime in µsec')
    ax4.set_ylabel('P(Solve)')
    ax4.set_xscale('log')

    ax5.plot(x_iter, y, linestyle='dotted')
    ax5.grid(b=True, which='both', color='#666666', linestyle='-', alpha=0.1)
    ax5.set_title('RLD (Search Steps) - Log-Log')
    ax5.set_xlabel('Iterations')
    ax5.set_ylabel('P(Solve)')
    ax5.set_xscale('log')
    ax5.set_yscale('log')

    ax6.plot(x_run, y, linestyle='dotted')
    ax6.grid(b=True, which='both', color='#666666', linestyle='-', alpha=0.1)
    ax6.set_title('RTD (Runtime) - Log-Log')
    ax6.set_xlabel('Runtime in µsec')
    ax6.set_ylabel('P(Solve)')
    ax6.set_xscale('log')
    ax6.set_yscale('log')

    plt.tight_layout()
    plt.savefig('./plot-rtd.png')
    plt.show()
