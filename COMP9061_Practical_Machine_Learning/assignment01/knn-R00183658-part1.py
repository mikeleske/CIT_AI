#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""knn-R00183658-part1.py: 
   
    COMP9061 - Practical Machine Learning Assignment 1
    k-NN for classification
"""

__author__      = "Mike Leske"
__copyright__   = "Copyright 2019, Mike Leske"


import numpy as np

# load the data
train = np.genfromtxt('./data/classification/trainingData.csv', delimiter=',')
test = np.genfromtxt('./data/classification/testData.csv', delimiter=',')

def calculateDistances(train, z):
    # Array operation to calculate euclidean distance from test instance to all training instances
    distance = np.sqrt(np.square(train - z).sum(axis = 1))
    return distance, np.argsort(distance)
    
def basic_knn(train, test, k=1):
    count = 0

    # Iterate over test data
    for i in range(test.shape[0]):

        # Split test instances into features and target
        X_test, y_test = test[i][:10], test[i][10]

        # Calculate euclidean distance
        # dist = distance to every point in traning data
        # idx  = sorted index (nearest to furthest)
        dist, idx = calculateDistances(train[:,:10], X_test)

        # Get target values of k closest points
        classes = train[idx[:k]][:, 10].astype(int)

        # Get the mode of k nearest classes as prediction
        mode = np.bincount(classes).argmax()

        # If mode equals test taret, increment count +1
        if mode == y_test: count += 1
    
    # Accuracy = number of correct predictions divided by total number of test items
    accuracy = count/test.shape[0]
    return accuracy

# Run the Basic KNN method
acc = basic_knn(train, test)
print(acc)