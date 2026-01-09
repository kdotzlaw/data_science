from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model, metrics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



if __name__=='__main__':
        
    # load Boston Housing dataset and read into data frame
    durl = "https://lib.stat.cmu.edu/datasets/boston"
    rdf = pd.read_csv(durl, sep="\s+",skiprows=22,header=None)

    # preprocess data by extracting x (input) and y (target) from dataframe
    x = np.hstack([rdf.values[::2,:],
                  rdf.values[1::2,:2]])
    y = rdf.values[1::2,2]

    # split data into training and test data
    trainX, testX, trainY, testY = train_test_split(x,y,test_size=0.4,random_state=1)

    # init linear regression model
    lr = linear_model.LinearRegression()

    # fit model to training data
    lr.fit(trainX,trainY)

    # determine regression coefficients (closer to 1 the better)
    print(f"Regression coefficients: {lr.coef_} ")

    # determine varience score (closer to 1 the better)
    # .score calcs varience by: 1-Varience(y_actual - y_pred)/varience(y_actual)
    print(f"Varience score: {lr.score(testX,testY)}")

    # plotting errors of predicted & actual values
    plt.style.use('fivethirtyeight')
    # errors in training data
    plt.scatter(lr.predict(trainX),lr.predict(trainX)-trainY, color='green',s=10,label='Training Data')
    # errors in test data
    plt.scatter(lr.predict(testX),lr.predict(testX)-testY, color="blue", s=10, label="Test Data")

    # plot line of 0 error
    plt.hlines(y=0,xmin=0,xmax=50,linewidth=2)

    # legend & plot meta
    plt.legend(loc="upper right")
    plt.title("Residual Errors")
    plt.show()