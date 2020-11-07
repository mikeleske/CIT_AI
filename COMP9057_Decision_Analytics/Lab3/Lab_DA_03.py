from ortools.sat.python import cp_model
import pandas as pd

def main():
    data = pd.read_excel("Lab03_data.xlsx", sheet_name=None)
    containers = data["Containers"]
    items = data["Items"]
            
    model = cp_model.CpModel()
    
    # one Boolean decision variable for every combination of item/bin
    x = {}
    for i in range(len(items)):
        for j in range(len(containers)):
            x[(i,j)] = model.NewBoolVar("x_%s_%s"%(i,j))
        
    # every item is at most in one bin
    for i in range(len(items)):    
        model.Add(sum([x[(i,j)] 
                       for j in range(len(containers))])<=1)
            
    # every bin's capacity is not exceeded
    for j in range(len(containers)):
        model.Add(sum([items.iloc[i]["Weight"]*x[(i,j)] 
                       for i in range(len(items))])<=containers.iloc[j]["Maximum capacity"])

    model.Maximize(sum([items.iloc[i]["Value"]*x[(i,j)] 
                        for i in range(len(items)) for j in range(len(containers))]))
           
    solver = cp_model.CpSolver()    
    status = solver.Solve(model)
    print(solver.StatusName(status))

    result_items = pd.DataFrame(index=items.index, columns=["Id", "Weight", "Value", "Container"])
    for i in range(len(items)):
        result_items["Id"][i] = items.iloc[i]["Id"]
        result_items["Weight"][i] = items.iloc[i]["Weight"]
        result_items["Value"][i] = items.iloc[i]["Value"]
        for j in range(len(containers)):
            if solver.Value(x[(i,j)]):
                result_items["Container"][i] = containers.iloc[j]["Id"]
                break

    result_containers = pd.DataFrame(index=containers.index, columns=["Id", "Maximum capacity", 
                                                                      "Weight (total)", 
                                                                      "Weight (percentage)", 
                                                                      "Value (total)", 
                                                                      "Value (percentage)"])
    for j in range(len(containers)):
        result_containers["Id"][j] = containers.iloc[j]["Id"]
        result_containers["Maximum capacity"][j] = containers.iloc[j]["Maximum capacity"]
        weight = 0
        value = 0
        for i in range(len(items)):
            if solver.Value(x[(i,j)]):
                weight += items.iloc[i]["Weight"]
                value += items.iloc[i]["Value"]
        result_containers["Weight (total)"][j] = weight
        result_containers["Weight (percentage)"][j] = 100*weight/containers.iloc[j]["Maximum capacity"]
        result_containers["Weight (total)"][j] = weight
        result_containers["Value (total)"][j] = value
        result_containers["Value (percentage)"][j] = 100*value/solver.BestObjectiveBound()

    writer = pd.ExcelWriter("Lab03_output.xlsx")    
    result_items.to_excel(writer, sheet_name="Items", index=False)
    result_containers.to_excel(writer, sheet_name="Containers", index=False)
    writer.close()

    print("Total item value", sum(items["Value"]))
    print("Total packed value", solver.BestObjectiveBound())
    print("Percentage", 100*solver.BestObjectiveBound()/sum(items["Value"]))
    
main()
