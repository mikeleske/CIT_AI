from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ortools.sat.python import cp_model

def SearchForOptimalSolution():
    model = cp_model.CpModel()
    
    knapsack_size = 15
    items = [(12,4),(2,2),(1,2),(1,1),(4,10)]

    in_knapsack = []
    weight_in_knapsack = []
    value_in_knapsack = []
    for i in range(0,len(items)):        
        in_knapsack.append(model.NewBoolVar("item_"+str(i)))
        weight_in_knapsack.append(items[i][0] * in_knapsack[i])
        value_in_knapsack.append(items[i][1] * in_knapsack[i])

    print('weight_in_knapsack:', weight_in_knapsack)
    print('value_in_knapsack:', value_in_knapsack)

    total_weight = sum(weight_in_knapsack)
    model.Add(total_weight <= knapsack_size)
    
    total_value = sum(value_in_knapsack)

    solver = cp_model.CpSolver()    

    model.Maximize(total_value)    
    status = solver.Solve(model)
    print(solver.StatusName(status))
   
    for i in range(0,len(items)):        
        if solver.Value(in_knapsack[i]):            
            print("Pack item "+str(i)+" (weight="+str(items[i][0])+",value="+str(items[i][1])+")")
    print("Total weight: "+str(solver.Value(total_weight)))
    print("Total value: "+str(solver.Value(total_value)))
    

SearchForOptimalSolution()