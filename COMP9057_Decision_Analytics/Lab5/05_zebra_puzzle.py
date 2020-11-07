from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ortools.sat.python import cp_model

class SolutionPrinter(cp_model.CpSolverSolutionCallback):
    """Print intermediate solutions."""

    def __init__(self, variables):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__variables = variables
        self.__solution_count = 0

    def on_solution_callback(self):
        print('on_solution_', self.__solution_count)
        self.__solution_count += 1
        #for v in self.__variables:
            #print(v)
            #print('%s=%i' % (v, self.Value(v)), end=' ')
        print()

    def solution_count(self):
        return self.__solution_count


def main():

    model = cp_model.CpModel()

    houses = ['House #1', 'House #2', 'House #3', 'House #4', 'House #5']

    colours = ['red', 'green', 'ivory', 'yellow', 'blue']
    house_colour = {}
    for house in houses:
        variables = {}
        for colour in colours:
            variables[colour] = model.NewBoolVar(house+colour)
        house_colour[house] = variables


    nationalities = ['English', 'Spanish', 'Ukranian', 'Norwegian', 'Japanese']
    house_nationality = {}
    for house in houses:
        variables = {}
        for nationality in nationalities:
            variables[nationality] = model.NewBoolVar(house+nationality)
        house_nationality[house] = variables
    

    pets = ['dog', 'snails', 'fox', 'horse', 'zebra']
    house_pet = {}
    for house in houses:
        variables = {}
        for pet in pets:
            variables[pet] = model.NewBoolVar(house+pet)
        house_pet[house] = variables
    

    drinks = ['coffee', 'tea', 'milk', 'juice', 'water']
    house_drink = {}
    for house in houses:
        variables = {}
        for drink in drinks:
            variables[drink] = model.NewBoolVar(house+drink)
        house_drink[house] = variables
    

    cigarettes = ['Old Gold', 'Chesterfields', 'Kools', 'Lucky Strike', 'Parliaments']
    house_cigarette = {}
    for house in houses:
        variables = {}
        for cigarette in cigarettes:
            variables[cigarette] = model.NewBoolVar(house+cigarette)
        house_cigarette[house] = variables

    
    for house in houses:
        model.AddBoolAnd([house_colour[house]['red']]).OnlyEnforceIf(house_nationality[house]['English'])


    for house in houses:
        model.AddBoolAnd([house_pet[house]['dog']]).OnlyEnforceIf(house_nationality[house]['Spanish'])


    for house in houses:
        model.AddBoolAnd([house_drink[house]['coffee']]).OnlyEnforceIf(house_colour[house]['green'])


    for house in houses:
        model.AddBoolAnd([house_drink[house]['tea']]).OnlyEnforceIf(house_nationality[house]['Ukranian'])


    for i in range(4):
        model.AddBoolAnd([house_colour[houses[i+1]]['green']]).OnlyEnforceIf(house_colour[houses[i]]['ivory'])
    model.AddBoolAnd([house_colour[houses[4]]['ivory'].Not()])
    model.AddBoolAnd([house_colour[houses[0]]['green'].Not()])


    for house in houses:
        model.AddBoolAnd([house_pet[house]['snails']]).OnlyEnforceIf(house_cigarette[house]['Old Gold'])
    

    for house in houses:
        model.AddBoolAnd([house_colour[house]['yellow']]).OnlyEnforceIf(house_cigarette[house]['Kools'])
    

    model.AddBoolAnd([house_drink['House #3']['milk']])


    model.AddBoolAnd([house_nationality['House #1']['Norwegian']])


    for i in range(1, 4):
        model.AddBoolOr([
                     house_pet[houses[i+1]]['fox'],
                     house_pet[houses[i-1]]['fox']]).OnlyEnforceIf(house_cigarette[houses[i]]['Chesterfields'])
    model.AddBoolOr([house_pet['House #2']['fox']]).OnlyEnforceIf(house_cigarette['House #1']['Chesterfields'])
    model.AddBoolOr([house_pet['House #4']['fox']]).OnlyEnforceIf(house_cigarette['House #5']['Chesterfields'])


    for i in range(1, 4):
        model.AddBoolOr([
                     house_pet[houses[i+1]]['horse'],
                     house_pet[houses[i-1]]['horse']]).OnlyEnforceIf(house_cigarette[houses[i]]['Kools'])
    model.AddBoolOr([house_pet['House #2']['horse']]).OnlyEnforceIf(house_cigarette['House #1']['Kools'])
    model.AddBoolOr([house_pet['House #4']['horse']]).OnlyEnforceIf(house_cigarette['House #5']['Kools'])


    for house in houses:
        model.AddBoolAnd([house_drink[house]['juice']]).OnlyEnforceIf(house_cigarette[house]['Lucky Strike'])
    

    for house in houses:
        model.AddBoolAnd([house_cigarette[house]['Parliaments']]).OnlyEnforceIf(house_nationality[house]['Japanese'])


    for i in range(1, 4):
        model.AddBoolOr([
                     house_nationality[houses[i+1]]['Norwegian'],
                     house_nationality[houses[i-1]]['Norwegian']]).OnlyEnforceIf(house_colour[houses[i]]['blue'])
    model.AddBoolOr([house_nationality[houses[1]]['Norwegian']]).OnlyEnforceIf(house_colour[houses[0]]['blue'])
    model.AddBoolOr([house_nationality[houses[3]]['Norwegian']]).OnlyEnforceIf(house_colour[houses[4]]['blue'])


    for house in houses:
        variables = []
        for colour in colours:
            variables.append(house_colour[house][colour])
        model.AddBoolOr(variables)
    

    for house in houses:
        variables = []
        for nationality in nationalities:
            variables.append(house_nationality[house][nationality])
        model.AddBoolOr(variables)


    for house in houses:
        variables = []
        for pet in pets:
            variables.append(house_pet[house][pet])
        model.AddBoolOr(variables)


    for house in houses:
        variables = []
        for drink in drinks:
            variables.append(house_drink[house][drink])
        model.AddBoolOr(variables)


    for house in houses:
        variables = []
        for cigarette in cigarettes:
            variables.append(house_cigarette[house][cigarette])
        model.AddBoolOr(variables)
    

    for house in houses:
        for i in range(5):
            for j in range(i+1, 5):
                model.AddBoolOr([
                    house_colour[house][colours[i]].Not(),
                    house_colour[house][colours[j]].Not()
                ])


    for house in houses:
        for i in range(5):
            for j in range(i+1, 5):
                model.AddBoolOr([
                    house_nationality[house][nationalities[i]].Not(),
                    house_nationality[house][nationalities[j]].Not()
                ])


    for house in houses:
        for i in range(5):
            for j in range(i+1, 5):
                model.AddBoolOr([
                    house_pet[house][pets[i]].Not(),
                    house_pet[house][pets[j]].Not()
                ])


    for house in houses:
        for i in range(5):
            for j in range(i+1, 5):
                model.AddBoolOr([
                    house_drink[house][drinks[i]].Not(),
                    house_drink[house][drinks[j]].Not()
                ])


    for house in houses:
        for i in range(5):
            for j in range(i+1, 5):
                model.AddBoolOr([
                    house_cigarette[house][cigarettes[i]].Not(),
                    house_cigarette[house][cigarettes[j]].Not()
                ])


    for i in range(5):
        for j in range(i+1, 5):
            for k in range(5):
                model.AddBoolOr([
                    house_colour[houses[i]][colours[k]].Not(),
                    house_colour[houses[j]][colours[k]].Not()
                ])


    for i in range(5):
        for j in range(i+1, 5):
            for k in range(5):
                model.AddBoolOr([
                    house_nationality[houses[i]][nationalities[k]].Not(),
                    house_nationality[houses[j]][nationalities[k]].Not()
                ])


    for i in range(5):
        for j in range(i+1, 5):
            for k in range(5):
                model.AddBoolOr([
                    house_pet[houses[i]][pets[k]].Not(),
                    house_pet[houses[j]][pets[k]].Not()
                ])


    for i in range(5):
        for j in range(i+1, 5):
            for k in range(5):
                model.AddBoolOr([
                    house_drink[houses[i]][drinks[k]].Not(),
                    house_drink[houses[j]][drinks[k]].Not()
                ])


    for i in range(5):
        for j in range(i+1, 5):
            for k in range(5):
                model.AddBoolOr([
                    house_cigarette[houses[i]][cigarettes[k]].Not(),
                    house_cigarette[houses[j]][cigarettes[k]].Not()
                ])

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 120
    #status = solver.Solve(model)
    solution_printer = SolutionPrinter(houses)
    status = solver.SearchForAllSolutions(model, solution_printer)
    
    print('\nSolver status:', solver.StatusName(status))
    
    #if solver.StatusName(status) == 'OPTIMAL' or solver.StatusName(status) == 'FEASIBLE':
    for house in houses:
        if solver.Value(house_drink[house]['water']):
            for nationality in nationalities:
                if solver.Value(house_nationality[house][nationality]):
                    print('The '+nationality+' drinks water.')
        if solver.Value(house_pet[house]['zebra']):
            for nationality in nationalities:
                if solver.Value(house_nationality[house][nationality]):
                    print('The '+nationality+' ows the zebra.')


main()

