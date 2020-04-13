# -*- coding: utf-8 -*-

from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd


def runClassifiers(feature_train, label_train, feature_test):
    
    nearestN = KNeighborsClassifier()
    nearestN.fit(feature_train, label_train)
    results = nearestN.predict(feature_test)
    
    return results

    

def performPreprocessing(titanic):

    
    # remove features
    titanic = titanic.drop(['Cabin'], axis=1)
    titanic = titanic.drop(['Ticket'], axis=1)
    titanic = titanic.drop(['Name'], axis=1)

    # impute the mean value for all missing values in the Fare and Age column   
    imputer = SimpleImputer(missing_values = np.NaN, strategy='mean')
    titanic[['Age', 'Fare']]= imputer.fit_transform(titanic[['Age', 'Fare']])

    # impute the most frequent value for the Embarked column
    imputer = SimpleImputer(missing_values = np.NaN, strategy='most_frequent')
    titanic[['Embarked']]= imputer.fit_transform(titanic[ ['Embarked']])

   
    # encode categorical variables as continuous variables    
    titanic['Gender'] = titanic['Sex'].map({'female': 0, 'male':1}).astype(int)
    titanic = titanic.drop(['Sex'], axis=1)  
    
    # one hot encode the Embarked column    
    encoder = OneHotEncoder(sparse=False)
    oneHotEncoded = encoder.fit_transform(titanic[["Embarked"]])
    dfOneHotEncoded = pd.DataFrame(oneHotEncoded )
    titanic = titanic.drop(['Embarked'], axis=1) 
    
    titanic = pd.concat([titanic, dfOneHotEncoded], axis=1)
    
 
   
    scalingObj = preprocessing.MinMaxScaler()
    titanic[["Age", "SibSp", "Parch", "Fare", "Pclass"]]= scalingObj.fit_transform(titanic[["Age", "SibSp", "Parch", "Fare", "Pclass"]])

    return titanic
    


def main():
    

    # Open the training dataset as a dataframe and perform preprocessing
    titanic_train = pd.read_csv("train.csv", delimiter=",")
    titanic_test = pd.read_csv("test.csv", delimiter=",")
    
    # Merge the two datasets into one dataframe    
    # Create a dummy survived column for the test data
    titanic_test["Survived"] = np.zeros(len(titanic_test))
    titanic_test["Survived"] = -1
    # Merge the two dataframes
    frameList = [titanic_train, titanic_test]
    allData = pd.concat(frameList, ignore_index=True, sort=True)
    
    
    # Run preprocessing.
    allData = performPreprocessing(allData)
    
    # Seperate the resulting data into test and train
    trainData = allData[allData["Survived"] != -1] 
    testData = allData[allData["Survived"] == -1]
    

    # Split the training dataset into features and classes
    label_train = trainData["Survived"]
    feature_train = trainData.drop(["Survived"], axis= 1)
   
    # Remove the passenger ID from training dataframe and testing data
    feature_train = feature_train.drop(['PassengerId'], axis=1)
    
    
    # Remove passenger ID from test data and store as a Series object
    passengerIDSeries = testData["PassengerId"].values
    feature_test = testData.drop(['PassengerId'], axis=1)
    
    # Drop the dummy survived column from the test data
    feature_test = feature_test.drop(["Survived"], axis= 1)
    
    results = runClassifiers(feature_train, label_train, feature_test)
    
    resultSeries = pd.Series(data = results, name = 'Survived', dtype='int64')
    
    df = pd.DataFrame({"PassengerId":passengerIDSeries, "Survived":resultSeries})

    df.to_csv("kaggle_submission.csv", index=False, header=True)
    



  
main()
