from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ortools.sat.python import cp_model


class VarArraySolutionPrinter(cp_model.CpSolverSolutionCallback):
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


def main(maxT=10):

    model = cp_model.CpModel()

    #
    # Predicate lists
    #
    WolfOnThisSide          = [ None for t in range(maxT)]
    SheepOnThisSide         = [ None for t in range(maxT)]
    CabbageOnThisSide       = [ None for t in range(maxT)]
    FerrymanOnThisSide      = [ None for t in range(maxT)]
    WolfOnOppositeSide      = [ None for t in range(maxT)]
    SheepOnOppositeSide     = [ None for t in range(maxT)]
    CabbageOnOppositeSide   = [ None for t in range(maxT)]
    FerrymanOnOppositeSide  = [ None for t in range(maxT)]

    for t in range(maxT):
        WolfOnThisSide[t]           = model.NewBoolVar('WolfOnThisSide'+str(t))
        SheepOnThisSide[t]          = model.NewBoolVar('SheepOnThisSide'+str(t))
        CabbageOnThisSide[t]        = model.NewBoolVar('CabbageOnThisSide'+str(t))
        FerrymanOnThisSide[t]       = model.NewBoolVar('FerrymanOnThisSide'+str(t))
        WolfOnOppositeSide[t]       = model.NewBoolVar('WolfOnOppositeSide'+str(t))
        SheepOnOppositeSide[t]      = model.NewBoolVar('SheepOnOppositeSide'+str(t))
        CabbageOnOppositeSide[t]    = model.NewBoolVar('CabbageOnOppositeSide'+str(t))
        FerrymanOnOppositeSide[t]   = model.NewBoolVar('FerrymanOnOppositeSide'+str(t))

    #
    # Operator lists
    #
    moveWolfAcross          = [ None for t in range(maxT)]
    moveSheepAcross         = [ None for t in range(maxT)]
    moveCabbageAcross       = [ None for t in range(maxT)]
    moveWolfBack            = [ None for t in range(maxT)]
    moveSheepBack           = [ None for t in range(maxT)]
    moveCabbageBack         = [ None for t in range(maxT)]
    moveAcross              = [ None for t in range(maxT)]
    moveBack                = [ None for t in range(maxT)]

    for t in range(maxT-1):
        moveWolfAcross[t]       = model.NewBoolVar('MoveWolfAcross'+str(t))
        moveSheepAcross[t]      = model.NewBoolVar('MoveSheepAcross'+str(t))
        moveCabbageAcross[t]    = model.NewBoolVar('MoveCabbageAcross'+str(t))
        moveWolfBack[t]         = model.NewBoolVar('MoveWolfBack'+str(t))
        moveSheepBack[t]        = model.NewBoolVar('MoveSheepBack'+str(t))
        moveCabbageBack[t]      = model.NewBoolVar('MoveCabbageBack'+str(t))
        moveAcross[t]           = model.NewBoolVar('MoveAcross'+str(t))
        moveBack[t]             = model.NewBoolVar('MoveBack'+str(t))
    
    #
    # Initial State
    #
    model.AddBoolAnd(
        [
            FerrymanOnThisSide[0],
            WolfOnThisSide[0],
            SheepOnThisSide[0],
            CabbageOnThisSide[0]
        ]
    )
    model.AddBoolAnd(
        [
            FerrymanOnOppositeSide[0].Not(),
            WolfOnOppositeSide[0].Not(),
            SheepOnOppositeSide[0].Not(),
            CabbageOnOppositeSide[0].Not()
        ]
    )
    
    #
    # Goal State
    #
    model.AddBoolAnd(
        [
            WolfOnOppositeSide[maxT-1],
            SheepOnOppositeSide[maxT-1],
            CabbageOnOppositeSide[maxT-1]
        ]
    )

    #
    # Pre/Post Conditions: across
    #
    for t in range(maxT-1):
        model.AddBoolAnd(
            [
                WolfOnThisSide[t], FerrymanOnThisSide[t],
                WolfOnOppositeSide[t+1], FerrymanOnOppositeSide[t+1],
                WolfOnThisSide[t+1].Not(), FerrymanOnThisSide[t+1].Not()
            ]
        ).OnlyEnforceIf(moveWolfAcross[t])

        model.AddBoolAnd(
            [
                SheepOnThisSide[t], FerrymanOnThisSide[t],
                SheepOnOppositeSide[t+1], FerrymanOnOppositeSide[t+1],
                SheepOnThisSide[t+1].Not(), FerrymanOnThisSide[t+1].Not()
            ]
        ).OnlyEnforceIf(moveSheepAcross[t])

        model.AddBoolAnd(
            [
                CabbageOnThisSide[t], FerrymanOnThisSide[t],
                CabbageOnOppositeSide[t+1], FerrymanOnOppositeSide[t+1],
                CabbageOnThisSide[t+1].Not(), FerrymanOnThisSide[t+1].Not()
            ]
        ).OnlyEnforceIf(moveCabbageAcross[t])

        model.AddBoolAnd(
            [
                FerrymanOnThisSide[t],
                FerrymanOnOppositeSide[t+1], FerrymanOnThisSide[t+1].Not(),
            ]
        ).OnlyEnforceIf(moveAcross[t])

    #
    # Pre/Post Conditions: back
    #
    for t in range(maxT-1):
        model.AddBoolAnd(
            [
                WolfOnOppositeSide[t], FerrymanOnOppositeSide[t],
                WolfOnThisSide[t+1], FerrymanOnThisSide[t+1],
                WolfOnOppositeSide[t+1].Not(), FerrymanOnOppositeSide[t+1].Not()
            ]
        ).OnlyEnforceIf(moveWolfBack[t])

        model.AddBoolAnd(
            [
                SheepOnOppositeSide[t], FerrymanOnOppositeSide[t],
                SheepOnThisSide[t+1], FerrymanOnThisSide[t+1],
                SheepOnOppositeSide[t+1].Not(), FerrymanOnOppositeSide[t+1].Not()
            ]
        ).OnlyEnforceIf(moveSheepBack[t])

        model.AddBoolAnd(
            [
                CabbageOnOppositeSide[t], FerrymanOnOppositeSide[t],
                CabbageOnThisSide[t+1], FerrymanOnThisSide[t+1],
                CabbageOnOppositeSide[t+1].Not(), FerrymanOnOppositeSide[t+1].Not()
            ]
        ).OnlyEnforceIf(moveCabbageBack[t])

        model.AddBoolAnd(
            [
                FerrymanOnOppositeSide[t],
                FerrymanOnThisSide[t+1], FerrymanOnOppositeSide[t+1].Not(),
            ]
        ).OnlyEnforceIf(moveBack[t])

    #
    # Frame Axioms
    #
    for t in range(maxT-1):
        model.AddBoolOr(
            [
                WolfOnThisSide[t+1].Not(), WolfOnThisSide[t], moveWolfBack[t]
            ]
        )

        model.AddBoolOr(
            [
                SheepOnThisSide[t+1].Not(), SheepOnThisSide[t], moveSheepBack[t]
            ]
        )

        model.AddBoolOr(
            [
                CabbageOnThisSide[t+1].Not(), CabbageOnThisSide[t], moveCabbageBack[t]
            ]
        )

        model.AddBoolOr(
            [
                WolfOnOppositeSide[t+1].Not(), WolfOnOppositeSide[t], moveWolfAcross[t]
            ]
        )

        model.AddBoolOr(
            [
                SheepOnOppositeSide[t+1].Not(), SheepOnOppositeSide[t], moveSheepAcross[t]
            ]
        )

        model.AddBoolOr(
            [
                CabbageOnOppositeSide[t+1].Not(), CabbageOnOppositeSide[t], moveCabbageAcross[t]
            ]
        )

        model.AddBoolOr(
            [
                FerrymanOnThisSide[t+1].Not(), FerrymanOnThisSide[t],
                moveWolfBack[t], moveSheepBack[t], moveCabbageBack[t], moveBack[t]
            ]
        )

        model.AddBoolOr(
            [
                FerrymanOnOppositeSide[t+1].Not(), FerrymanOnOppositeSide[t],
                moveWolfAcross[t], moveSheepAcross[t], moveCabbageAcross[t], moveAcross[t]
            ]
        )
    
    #
    # Exclusion Axioms
    #
    for t in range(maxT-1):
        model.AddBoolOr([moveWolfAcross[t].Not(), moveSheepAcross[t].Not()])
        model.AddBoolOr([moveWolfAcross[t].Not(), moveCabbageAcross[t].Not()])
        model.AddBoolOr([moveWolfAcross[t].Not(), moveAcross[t].Not()])
        model.AddBoolOr([moveWolfAcross[t].Not(), moveWolfBack[t].Not()])
        model.AddBoolOr([moveWolfAcross[t].Not(), moveSheepBack[t].Not()])
        model.AddBoolOr([moveWolfAcross[t].Not(), moveCabbageBack[t].Not()])
        model.AddBoolOr([moveWolfAcross[t].Not(), moveBack[t].Not()])

        model.AddBoolOr([moveSheepAcross[t].Not(), moveWolfAcross[t].Not()])
        model.AddBoolOr([moveSheepAcross[t].Not(), moveCabbageAcross[t].Not()])
        model.AddBoolOr([moveSheepAcross[t].Not(), moveAcross[t].Not()])
        model.AddBoolOr([moveSheepAcross[t].Not(), moveWolfBack[t].Not()])
        model.AddBoolOr([moveSheepAcross[t].Not(), moveSheepBack[t].Not()])
        model.AddBoolOr([moveSheepAcross[t].Not(), moveCabbageBack[t].Not()])
        model.AddBoolOr([moveSheepAcross[t].Not(), moveBack[t].Not()])

        model.AddBoolOr([moveCabbageAcross[t].Not(), moveWolfAcross[t].Not()])
        model.AddBoolOr([moveCabbageAcross[t].Not(), moveSheepAcross[t].Not()])
        model.AddBoolOr([moveCabbageAcross[t].Not(), moveAcross[t].Not()])
        model.AddBoolOr([moveCabbageAcross[t].Not(), moveWolfBack[t].Not()])
        model.AddBoolOr([moveCabbageAcross[t].Not(), moveSheepBack[t].Not()])
        model.AddBoolOr([moveCabbageAcross[t].Not(), moveCabbageBack[t].Not()])
        model.AddBoolOr([moveCabbageAcross[t].Not(), moveBack[t].Not()])

        model.AddBoolOr([moveAcross[t].Not(), moveWolfAcross[t].Not()])
        model.AddBoolOr([moveAcross[t].Not(), moveSheepAcross[t].Not()])
        model.AddBoolOr([moveAcross[t].Not(), moveCabbageAcross[t].Not()])
        model.AddBoolOr([moveAcross[t].Not(), moveWolfBack[t].Not()])
        model.AddBoolOr([moveAcross[t].Not(), moveSheepBack[t].Not()])
        model.AddBoolOr([moveAcross[t].Not(), moveCabbageBack[t].Not()])
        model.AddBoolOr([moveAcross[t].Not(), moveBack[t].Not()])
    
        model.AddBoolOr([moveBack[t].Not(), moveWolfBack[t].Not()])
        model.AddBoolOr([moveBack[t].Not(), moveSheepBack[t].Not()])
        model.AddBoolOr([moveBack[t].Not(), moveCabbageBack[t].Not()])
    #
    # Additional constraints
    #
    for t in range(maxT):
        model.AddBoolOr(
            [
                WolfOnThisSide[t].Not(), SheepOnThisSide[t].Not()
            ]
        ).OnlyEnforceIf(FerrymanOnThisSide[t].Not())

        model.AddBoolOr(
            [
                WolfOnOppositeSide[t].Not(), SheepOnOppositeSide[t].Not()
            ]
        ).OnlyEnforceIf(FerrymanOnOppositeSide[t].Not())

        model.AddBoolOr(
            [
                SheepOnThisSide[t].Not(), CabbageOnThisSide[t].Not()
            ]
        ).OnlyEnforceIf(FerrymanOnThisSide[t].Not())

        model.AddBoolOr(
            [
                SheepOnOppositeSide[t].Not(), CabbageOnOppositeSide[t].Not()
            ]
        ).OnlyEnforceIf(FerrymanOnOppositeSide[t].Not())

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 0.01
    #status = solver.Solve(model)
    solution_printer = VarArraySolutionPrinter([moveWolfAcross])

    status = solver.SearchForAllSolutions(model, solution_printer)
    print('\nSolver status:', solver.StatusName(status))

    #
    # Print Solution
    #
    for t in range(maxT-1):
        if solver.Value(moveWolfAcross[t]): print('t =', t, 'move wolf across')
        if solver.Value(moveWolfBack[t]): print('t =', t, 'move wolf back')
        if solver.Value(moveSheepAcross[t]): print('t =', t, 'move sheep across')
        if solver.Value(moveSheepBack[t]): print('t =', t, 'move sheep back')
        if solver.Value(moveCabbageAcross[t]): print('t =', t, 'move cabbage across')
        if solver.Value(moveCabbageBack[t]): print('t =', t, 'move cabbage back')
        if solver.Value(moveAcross[t]): print('t =', t, 'move across')
        if solver.Value(moveBack[t]): print('t =', t, 'move back')

    print('\nF S W C -- This Side')
    print('-------')
    for t in range(maxT-1):
        print(solver.Value(FerrymanOnThisSide[t]), solver.Value(SheepOnThisSide[t]), solver.Value(WolfOnThisSide[t]), solver.Value(CabbageOnThisSide[t]))

main(maxT = 8)