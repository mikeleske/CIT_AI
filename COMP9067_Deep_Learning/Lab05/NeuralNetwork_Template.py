
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model
from sklearn.metrics import accuracy_score



def logistic(x):
  return 1.0 / (1.0 + np.exp(-1.0*x))


def forwardPass(W1, b1, W2, b2, X):

    #TODO 2: Complete this function
    
    # push all traing data through the first hidden layer
    A1 = np.dot(W1, X) + b1
    H1 = np.tanh(A1)
    
    # next push all the outputs from the first layer (for each training example)
    # through the final layer, which is just a single Sigmoid unit
    A2 = np.dot(W2, H1) + b2
    H2 = logistic(A2)
    
    return H2, H1



def backward_propagation(X, Y, W2, H1, H2):  

    m = X.shape[1]
    
    dA2= H2 - Y
    dW2 = (1 / m) * np.dot(dA2, H1.T)
    db2 = (1 / m) * np.sum(dA2, axis=1, keepdims=True)
    dA1 = np.multiply(np.dot(W2.T, dA2), 1 - np.power(H1, 2))
    dW1 = (1 / m) * np.dot(dA1, X.T)
    db1 = (1 / m) * np.sum(dA1, axis=1, keepdims=True)
    
    
    return dW1, db1, dW2, db2



def calculateAccuracy(predictedYValues, Y,m):
    
    # If the probability is less than 0.5 set class to 0
    # If probability is greater than 0.5 set class to 1 

    predictedYValues = np.round(predictedYValues)
    acc = (accuracy_score(Y.flatten(), predictedYValues.flatten())  )
    return acc



def runLogisticRegression(X, Y):
    
    # Train the logistic regression classifier
    clf = sklearn.linear_model.LogisticRegressionCV(cv=10)
    clf.fit(X, Y)
    predictedY  = clf.predict(X)
    print ('Accuracy of logistic regression: ', accuracy_score(Y, predictedY), "(percentage of correctly labelled datapoints)")
 

def gradient_descent(X, Y, W1, b1, W2, b2, learning_rate):
    
    num_iterations = 4000
    numSamples = X.shape[1]
    
    for i in range(0, num_iterations):
         
        ### START CODE HERE ### (≈ 4 lines of code)
        # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".
        H2, H1= forwardPass(W1, b1, W2, b2, X)
        
        accuracy = calculateAccuracy(H2, Y, numSamples)
        
        if i%50 == 0:
            print (accuracy)
        
        # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
        dW1, db1, dW2, db2 = backward_propagation(X, Y, W2, H1, H2)
 
        # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".

        
        # TODO 3: Adjust the following to implement your gradient descent update rule below. 
        W1 = W1 - learning_rate * dW1
        b1 = b1 - learning_rate * db1
        W2 = W2 - learning_rate * dW2
        b2 = b2 - learning_rate * db2
        
    return W1, b1, W2, b2


def runNeuralNetwork(X, Y):
    
    X = X.T
    Y = Y.reshape(1, -1)
    
    # View dataset as a scatter plot
    #plt.scatter(X[0, :], X[1, :], c=Y.flatten(), s=40, cmap=plt.cm.Spectral)
    #plt.show()
    # Our training data must be in a matrix 
    # where each row is a feature and each column is an instance
   
    numFeatures = X.shape[0] # size of input layer
    numHidNeurons = 4
    numOutputUnits = 1
    learning_rate = 0.3
    
    # Initialize neural network weights and bias values
    np.random.seed(2)
    
    # TODO 1: Specify the correct shape for the W1 and W2 matrix
    # and for b1 and b2
    W1 = np.random.randn(numHidNeurons, numFeatures) * 0.01
    b1 = np.zeros(shape=(numHidNeurons, 1))
    W2 = np.random.randn(numOutputUnits, numHidNeurons) * 0.01
    b2 = np.zeros(shape=(numOutputUnits, 1))    

    W1, b1, W2, b2 = gradient_descent(X, Y, W1, b1, W2, b2, learning_rate)
    
    predictedYProb, H1 = forwardPass(W1, b1, W2, b2, X)
    predictedYValues = np.round(predictedYProb)

    # Visualize the predicted y values 
    plt.scatter(X[0, :], X[1, :], c=predictedYValues.flatten(), s=40, cmap=plt.cm.Spectral)
    plt.show()
       

def main():

    np.random.seed(1)
    num_Samples = 300
    
    # The dataset we are going to use is a basic circles dataset. 
    # The X value is the feature data and the Y array are the class values labels. 
    # Note the training data just contains two features (x1, x2)
    X,Y  = sklearn.datasets.make_circles(n_samples=num_Samples, factor=.5, noise=.15)
   
    # First we run logistic regression to see how it works on this problem
    #runLogisticRegression(X, Y)

    # Now build and train a neural network to solve this problem
    runNeuralNetwork(X, Y)
 

main()





