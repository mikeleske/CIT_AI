#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""knn-R00183658-part2.py: 
   
    COMP9061 - Practical Machine Learning Assignment 1 - Part 2
    weighted k-NN for classification problems
"""

__author__      = "Mike Leske"
__copyright__   = "Copyright 2019, Mike Leske"


import numpy as np
from collections import defaultdict

# load the data
train = np.genfromtxt('./data/classification/trainingData.csv', delimiter=',')
test = np.genfromtxt('./data/classification/testData.csv', delimiter=',')

def calculateDistances(train, z):
    # Array operation to calculate euclidean distance from test instance to all training instances
    distance = np.sqrt(np.square(train - z).sum(axis = 1))
    return distance, np.argsort(distance)

def weighted_knn_clf(train, test, k, n):
    # Initialize counter for correct classifications
    count = 0

    # Iterate over test data
    for i in range(test.shape[0]):
        # Initialize dict to calculate per-class weighted prediction
        vote = defaultdict()
        # Get in instance features and target
        X, y = test[i][:10], test[i][10]

        # Calculate euclidean distance
        # dist = distance to every point in traning data
        # idx  = sorted index (nearest to furthest)
        dist, idx = calculateDistances(train[:,:10], X)
        
        # Get the classes of the k nearest neighbors
        classes = train[idx[:k]][:, 10].astype(int)

        # Initialize classes in dict
        for c in classes:
            vote[c] = 0
        
        # For each k neighbors add the inverse (squared) distance to class dict entry
        for s in range(k):
            vote[train[idx[s]][10]] += 1/(np.power(dist[s], n))

        # If best prediction equals test target, increase counter
        if max(vote, key=vote.get) == y : count += 1
    
    # Calculate accuracy
    accuracy = count/test.shape[0]
    return accuracy

r2 = weighted_knn_clf(train, test, k=10, n=2)
print(r2)