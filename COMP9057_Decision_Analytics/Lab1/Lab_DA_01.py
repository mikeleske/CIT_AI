from ortools.sat.python import cp_model

class SolutionPrinter(cp_model.CpSolverSolutionCallback):
    def __init__(self, items, in_knapsack, total_weight, total_value):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.items_ = items
        self.in_knapsack_ = in_knapsack
        self.total_weight_ = total_weight
        self.total_value_ = total_value

    def OnSolutionCallback(self):        
        print("Feasible solution:")
        for i in range(0,len(self.items_)):        
            if self.Value(self.in_knapsack_[i]):            
                print("  - Pack item "+str(i)+" (weight="+str(self.items_[i][0])+",value="+str(self.items_[i][1])+")")
        print("  - Total weight: "+str(self.Value(self.total_weight_)))
        print("  - Total value: "+str(self.Value(self.total_value_)))


def main():
    model = cp_model.CpModel()
    
    knapsack_size = 15
    items = [(12,4),(2,2),(2,1),(1,1),(4,10)]
    
    # knapsack_size = 10
    # items = [(9,1),(2,1),(3,1)]
 
    in_knapsack = []
    weight_in_knapsack = []
    value_in_knapsack = []
    for i in range(0,len(items)):        
        in_knapsack.append(model.NewBoolVar("item_"+str(i)))
        weight_in_knapsack.append(items[i][0] * in_knapsack[i])
        value_in_knapsack.append(items[i][1] * in_knapsack[i])

    total_weight = sum(weight_in_knapsack)
    print(weight_in_knapsack)
    model.Add(total_weight <= knapsack_size)
    
    total_value = sum(value_in_knapsack)

    solver = cp_model.CpSolver()    
    solver.SearchForAllSolutions(model, SolutionPrinter(items, in_knapsack, total_weight, total_value))

    print()
    model.Maximize(total_value)    
    status = solver.Solve(model)
    print(solver.StatusName(status))
   
    for i in range(0,len(items)):        
        if solver.Value(in_knapsack[i]):            
            print("Pack item "+str(i)+" (weight="+str(items[i][0])+",value="+str(items[i][1])+")")
    print("Total weight: "+str(solver.Value(total_weight)))
    print("Total value: "+str(solver.Value(total_value)))
    
    
main()
