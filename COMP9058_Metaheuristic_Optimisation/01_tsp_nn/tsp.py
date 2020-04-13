#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""tsp.py: TSP solution generator

    In this lab we implement the following heuristic solution for the TSP: Nearest neighbor insertion
    - Start with randomly selected city and insert each new city into the current tour after the city to which it is closest.
    - (If there is more than one city to which it is closet, insert it after the first such city you encounter).
"""

__author__      = "Mike Leske"
__copyright__   = "Copyright 2019"

import numpy as np
import random
import sys, os
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('Qt5Agg')

def load_data(f):
    count = np.loadtxt(f, delimiter=' ', max_rows=1)
    cities = np.loadtxt(f, delimiter=' ', dtype={'names': ('id', 'x', 'y'), 'formats': ('int', 'f4', 'f4')}, skiprows=1)
    return count, cities

def find_path(count, cities):
    idx = int(random.randint(1, count))
    not_visited = cities[cities['id'] != idx]['id'].tolist()
    tsp_arr = np.array([[idx, 0]])

    while not_visited:
        cur_x = cities[cities['id'] == idx]['x']
        cur_y = cities[cities['id'] == idx]['y']
        remaining_cities = cities[np.isin(cities['id'], not_visited)]
        distance = np.array([
            remaining_cities['id'],
            np.around(np.sqrt(np.power(remaining_cities['x'] - cur_x, 2) + np.power(remaining_cities['y'] - cur_y, 2)))
        ])
        tsp_arr = np.append(tsp_arr, np.array([[distance[0][distance[1].argmin()], distance[1].min()]]), axis=0)
        #print('Nearest neighbor to {} =  {}  -  {}'.format(idx, distance[0][distance[1].argmin()], distance[1].min()))
        not_visited.remove(distance[0][distance[1].argmin()])
        idx = int(distance[0][distance[1].argmin()])

    #
    # Add the patch back to initial idx
    #
    start = tsp_arr[0][0]
    final_dist = np.around(np.sqrt(np.power(cities[cities['id'] == start]['x'] - cities[cities['id'] == idx]['x'], 2) + np.power(cities[cities['id'] == start]['y'] - cities[cities['id'] == idx]['y'], 2)))
    tsp_arr = np.append(tsp_arr, np.array([[start, final_dist]]), axis=0)
    
    return tsp_arr

def print_solution(tsp_arr):
    print('Path: {}'.format(tsp_arr[:, 0].astype(int)))
    print('Cost: {}'.format(tsp_arr[:, 1].sum().astype(int)))
    
def write_output(f, tsp_arr):
    with open(f, "wb") as f:
        np.savetxt(f, tsp_arr[:, 1].sum().astype(int), fmt='%i')
        np.savetxt(f, tsp_arr[:-1, 0].astype(int), fmt='%i')

def viz_path(tsp_arr, cities):
    fig, ax = plt.subplots(figsize=(20, 20))

    x = []
    y = []

    for i in tsp_arr[:, 0]:
        i = int(i) - 1
        x.append(cities[i]['x'])
        y.append(cities[i]['y'])

    plt.plot(x, y, color='green', marker='o', linestyle='dashed', linewidth=1, markersize=2)
    plt.show()
    
def main(argv):
    in_file = argv[1]
    out_file = argv[2]
    count, cities = load_data(in_file)
    tsp_arr = find_path(count, cities)
    #print_solution(tsp_arr)
    write_output(out_file, tsp_arr)
    viz_path(tsp_arr, cities)
    
if __name__ == "__main__":
    if len(sys.argv) > 3:
        print('Insufficient parameters provided. Sample: bash$ python tsp.py instance.tsp sol.tsp')
        exit()
    elif not os.path.exists(sys.argv[1]):
        print('ERROR: The input file does not exist.')
    else:
        main(sys.argv)
        
