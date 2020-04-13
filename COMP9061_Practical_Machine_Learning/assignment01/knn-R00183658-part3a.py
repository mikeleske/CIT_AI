#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""knn-R00183658-part3a.py: 
   
    COMP9061 - Practical Machine Learning Assignment 1 - Part 3a
    weighted k-NN for regression problems
"""

__author__      = "Mike Leske"
__copyright__   = "Copyright 2019, Mike Leske"


import numpy as np
import random

# load the data
train = np.genfromtxt('./data/regression/trainingData.csv', delimiter=',')
test = np.genfromtxt('./data/regression/testData.csv', delimiter=',')

def calculateDistances(train, z):
    # Array operation to calculate euclidean distance from test instance to all training instances
    distance = np.sqrt(np.square(train - z).sum(axis = 1))
    return distance, np.argsort(distance)

def weighted_knn_reg(train, test, k, n):
    # Split train and test instances into features and target
    X_train, y_train = train[:, :12], train[:, 12]
    X_test, y_test = test[:, :12], test[:, 12]
    
    # Calculate the mean test target
    y_mean = y_test.sum()/len(y_test)

    # Initialize nominator and denominator for r2 calculation
    sum_sq_res = 0
    tot_sum_sq = 0
    
    # Iterate over test data
    for i in range(X_test.shape[0]):
        # Get in instance features and target
        X, y = X_test[i], y_test[i]

        # Calculate euclidean distance
        # dist = distance to every point in traning data
        # idx  = sorted index (nearest to furthest)
        dist, idx = calculateDistances(X_train, X)
        
        w_sum = 0
        w_div = 0
        # Iterate over k nearest neighbors
        for s in range(k):
            # Calculate the inverse squared distance
            inv_dist = 1/np.power(dist[idx[s]], n)
            # Update weighted nominator and denominator
            w_sum += inv_dist * y_train[idx[s]]
            w_div += inv_dist

        # Calculate prediction for ith test instance
        pred = w_sum/w_div

        # Update nominator and denominator for r2 calculation
        sum_sq_res += np.square(y - pred)
        tot_sum_sq += np.square(y - y_mean)

    # Calculate overall r2 for test data
    r_squared = 1-(sum_sq_res/tot_sum_sq)
    return r_squared

r2 = weighted_knn_reg(train, test, k=11, n=2)
print(r2)