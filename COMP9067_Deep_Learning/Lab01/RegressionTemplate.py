   
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def linearRegression(X, Y):
    
    # set initial parameters for model
    bias = 0
    lambda1 = 0
    
    alpha = 0.2 # learning rate
    max_iter = 50

    #TODO
    # call gredient decent to calculate intercept(=bias) and slope(lambda1)
    bias, lambda1, history = gradient_descent(bias, lambda1, alpha, X, Y, max_iter)
    print ('Final bias and  lambda1 values are = ', bias, lambda1, " respecively." )
    
    # plot the data and overlay the linear regression model
    yPredictions = (lambda1*X)+bias
    plt.scatter(X, Y)
    plt.plot(X,yPredictions,'k-')
    plt.show()

    # plot the loss history
    plt.plot(range(len(history)), history,'k-')
    plt.show()


def gradient_descent(bias, lambda1, alpha, X, Y, max_iter):

    history = []
    for _ in range(max_iter):
        pred = lambda1*X + bias
        loss = 1/(2*X.shape[0]) * np.sum((pred - Y)**2)
        history.append(loss)

        # update lambda1 and bias
        lambda1 = lambda1 - alpha * 1/(2*X.shape[0]) * np.sum((pred - Y) * X)
        bias  = bias - alpha * 1/(2*X.shape[0]) * np.sum(pred - Y)

    return bias, lambda1, history
    
    
def main():
    
    # Read data into a dataframe
    df = pd.read_excel('data.xlsx')
    df = df.dropna() 

    # Convert Dataframe to a NumPy array
    X = df.values
    plt.scatter(X[:,1], X[:,0])
    plt.show()

    # Store feature and target data in separate arrays
    Y = df['Y'].values
    X = df['X'].values
    
    # Perform standarization on the feature data
    X = (X - np.mean(X))/np.std(X)
    
    linearRegression(X, Y)
    

main()
