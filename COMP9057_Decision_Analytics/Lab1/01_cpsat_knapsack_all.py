from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ortools.sat.python import cp_model


class VarArraySolutionPrinter(cp_model.CpSolverSolutionCallback):
    """Print intermediate solutions."""

    def __init__(self, variables, items=None):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__variables = variables
        self.__solution_count = 0
        self.__items = items

    def on_solution_callback(self):
        self.__solution_count += 1
        sum_w = 0
        sum_v = 0
        for v in self.__variables:
            print('%s=%i' % (v, self.Value(v)), end=' ')
            if self.Value(v):
                sum_w += self.__items[str(v)]['w']
                sum_v += self.__items[str(v)]['v']
        #print()
        print('w=%i v=%i' % (sum_w, sum_v))

    def solution_count(self):
        return self.__solution_count


def SearchForAllSolutionsSampleSat(optimal=False):
    """Showcases calling the solver to search for all solutions."""
    # Creates the model.
    model = cp_model.CpModel()

    items = {
        'b1': {'w': 1, 'v': 1},
        'b2': {'w': 1, 'v': 2},
        'b3': {'w': 2, 'v': 2},
        'b4': {'w': 4, 'v': 10},
        'b5': {'w': 12, 'v': 4},
    }

    # Creates the variables.
    MAX_WEIGHT = 15

    b1 = model.NewBoolVar('b1')
    b2 = model.NewBoolVar('b2')
    b3 = model.NewBoolVar('b3')
    b4 = model.NewBoolVar('b4')
    b5 = model.NewBoolVar('b5')

    # Create the constraints.
    model.Add(
        items['b1']['w']*b1 + 
        items['b2']['w']*b2 + 
        items['b3']['w']*b3 + 
        items['b4']['w']*b4 + 
        items['b5']['w']*b5 <= MAX_WEIGHT)

    # Create a solver and solve.
    solver = cp_model.CpSolver()
    solution_printer = VarArraySolutionPrinter([b1, b2, b3, b4, b5], items)
    
    status = solver.SearchForAllSolutions(model, solution_printer)

    print('Status = %s' % solver.StatusName(status))
    print('Number of solutions found: %i' % solution_printer.solution_count())


SearchForAllSolutionsSampleSat()