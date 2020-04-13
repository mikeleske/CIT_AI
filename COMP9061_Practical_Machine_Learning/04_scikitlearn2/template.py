# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np




def performPreprocessing(titanic):

    #TODO: Complete preprocessing here
    return 0
    


def main():
    

    # Open the training dataset as a dataframe and perform preprocessing
    titanic_train = pd.read_csv("train.csv", delimiter=",")
    titanic_test = pd.read_csv("test.csv", delimiter=",")

    
    # Merge the two datasets into one dataframe    
    titanic_test["Survived"] = np.zeros(len(titanic_test))
    titanic_test["Survived"] = -1
    frameList = [titanic_train, titanic_test]
    allData = pd.concat(frameList, ignore_index=True, sort='True')
    
    
    # Run preprocessing. Seperate the resulting data into test and train
    allData = performPreprocessing(allData)

  
main()
