from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ortools.sat.python import cp_model
import pandas as pd


def load_data(io):
    containers = pd.read_excel(io, sheet_name='Containers')
    items = pd.read_excel(io, sheet_name='Items')
    return (containers, items)


def main(io):
    model = cp_model.CpModel()

    containers, items = load_data(io)

    max_value = sum(items.Value)

    bins = {}

    for container in containers.iterrows():
        cid = container[1].Id
        bins[cid] = {}
        bins[cid]['capacity'] = container[1]['Maximum capacity']
        bins[cid]['in_knapsack'] = []
        bins[cid]['weight_in_knapsack'] = []
        bins[cid]['value_in_knapsack'] = []

        for item in items.iterrows():
            i = item[0]
            item_id = item[1].Id
            bins[cid]['in_knapsack'].append(model.NewBoolVar("item_"+str(cid)+"_"+str(item_id)))
            bins[cid]['weight_in_knapsack'].append(item[1].Weight * bins[cid]['in_knapsack'][i])
            bins[cid]['value_in_knapsack'].append(item[1].Value * bins[cid]['in_knapsack'][i])

    cids = bins.keys()
    
    for cid in cids:
        model.Add(sum(bins[cid]['weight_in_knapsack']) <= bins[cid]['capacity'])
    
    for i in range(0, len(items)):
        model.Add(bins['A']['in_knapsack'][i] == 0).OnlyEnforceIf(bins['B']['in_knapsack'][i])
        model.Add(bins['B']['in_knapsack'][i] == 0).OnlyEnforceIf(bins['A']['in_knapsack'][i])

    total_weight = sum([sum(v['weight_in_knapsack']) for k,v in bins.items()])
    total_value = sum([sum(v['value_in_knapsack']) for k,v in bins.items()])

    model.Maximize(total_value)

    solver = cp_model.CpSolver()

    status = solver.Solve(model)
    print('\nSolver status:', solver.StatusName(status))

    for k,v in bins.items():
        print('Container:', k)
        for i in range(0,len(v['in_knapsack'])):
            if solver.Value(v['in_knapsack'][i]):
                 print("  -> Pack item "+str(i)+" (weight={}, value={}".format(
                     str(bins[k]['weight_in_knapsack'][i]),
                     str(bins[k]['value_in_knapsack'][i])
                 ))

    print("Total weight: "+str(solver.Value(total_weight)))
    print("Total value: {} / {}%".format(
        str(solver.Value(total_value)),
        round(solver.Value(total_value)/max_value * 100, 1)
    ))

main('Lab03_data.xlsx')