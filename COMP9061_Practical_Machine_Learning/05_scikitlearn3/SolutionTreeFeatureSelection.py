# -*- coding: utf-8 -*-
"""

@author: Ted.Scully
"""

from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics


# Solution to Random Tree Question 



def runModel(train_features, train_labels, test_features, test_labels):
        # create an instance based learner
    clf = KNeighborsClassifier()
    clf = clf.fit(train_features, train_labels)
    
    # predict the class for an unseen example
    results= clf.predict(test_features)
    return metrics.accuracy_score(results, test_labels)
    

def main():
    
    # open the data file 
    allData = np.genfromtxt("dataFile.csv", delimiter=",")
    
    # split the data into training and class data
    y = allData[:, -1]
    X = allData[:, 0:-1]
    
    # standarize all data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    train_features, test_features, train_labels, test_labels = model_selection.train_test_split( X, y, test_size=0.2, random_state=1)
   
    accur = runModel(train_features, train_labels, test_features, test_labels)
    print ("Initial Accuracy ", accur)
    
    
    # create a extra tree classifier to be used for feature selection
    forest = RandomForestClassifier(n_estimators=250, random_state=0)
    forest.fit(X, y)
    
    # importances contains the feature importance value for each metric
    importances = forest.feature_importances_
    # argsort returns the indices that will sort the original array
    sortedIndices = np.argsort(importances)
   
    
    # iteratively remove the features with the lowest ranking and record accuracy
    numberOfFeatures = []
    accurKNN = []
    
    for num in range(0, 39):
        
        
        numberOfFeatures.append(num)
        
        # obtain indices to delete by slicing the order indices from argsort function
        indicesToDelete = sortedIndices[0:num+1]
        # delete identified indices
        train_features_new = np.delete(train_features, indicesToDelete, axis = 1)
        test_features_new = np.delete(test_features, indicesToDelete, axis = 1)
        
        # run the model using updates feature training and test data
        accur = runModel(train_features_new, train_labels, test_features_new, test_labels)
        accurKNN.append(accur)
    
    
    plt.figure()
    plt.xlabel("Number of features removed")
    plt.ylabel("Cross validation score ")
    plt.plot(numberOfFeatures, accurKNN)
    plt.show()
    

main()
    
