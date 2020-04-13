
    
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



# This function will take in all the feature data X
# as well as the current coefficient and bias values
# It should multiply all the feature value by their associated 
# coefficient and add the bias. It should then return the predicted 
# y values
def hypothesis(X, coefficients, bias):
    h = coefficients.dot(X.T) + bias
    return h


def calculateRSquared(bias, coefficients,X, Y):
    predictedY = hypothesis(X, coefficients, bias)
    
    avgY = np.average(Y)
    totalSumSq = np.sum((avgY - Y)**2)
    sumSqRes = np.sum((predictedY - Y)**2)
    r2 = 1.0-(sumSqRes/totalSumSq)
    
    return r2


def gradient_descent(bias, coefficients, alpha, X, Y, max_iter):
    
    length = len(X)
    
    # array is used to store change in cost function for each iteration of GD
    errorValues = []
    
    for num in range(0, max_iter):
        
        # TODO: 
        # Calculate predicted y values for current coefficient and bias values 
        # calculate and update bias using gradient descent rule
        # Update each coefficient value in turn using gradient descent rule
        h = hypothesis(X, coefficients, bias)
        loss = h - Y
        cost = np.sum(loss**2) / (length + 2)
        errorValues.append(cost)
        
        d_bias = np.sum(loss) / (length * 2)
        d_coeff = X.T.dot(loss.T) / (length * 2)

        coefficients -= alpha * d_coeff
        bias -= alpha * d_bias

    # calculate R squared value for current coefficient and bias values
    rSquared = calculateRSquared(bias, coefficients, X, Y)
    print ("Train R2 value is ", rSquared)

    # plot the cost for each iteration of gradient descent
    plt.plot(errorValues)
    plt.show()
    
    return bias, coefficients


def multipleLinearRegression(train, test):
    X_train, y_train = train
    X_test, y_test = test

    # set the number of coefficients equal to the number of features
    coefficients = np.zeros(X_train.shape[1])
    bias = 0.0
    alpha = 0.1
    max_iter=100

    # call gredient decent, and get intercept(=bias) and coefficents
    bias, coefficients = gradient_descent(bias, coefficients, alpha, X_train, y_train, max_iter)
    
    # Calculate rsquared on test set
    rSquared = calculateRSquared(bias, coefficients, X_test, y_test)
    print ("Test R2 value is ", rSquared)


def main():
    df_train = pd.read_csv("Dataset/trainingData.csv")
    df_test = pd.read_csv("Dataset/testData.csv")

    data_train = df_train.values
    data_test = df_test.values

    # Seperate the features from the target feature    
    train = (data_train[:, :-1], data_train[:, -1])
    test = (data_test[:, :-1], data_test[:, -1])
     
    # run regression function
    multipleLinearRegression(train, test)
    

main()
