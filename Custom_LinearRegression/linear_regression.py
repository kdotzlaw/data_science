import numpy as np
import matplotlib.pyplot as plt

'''
INPUT: x (independent variable), y (dependant variable)
OUTPUT:
PROCESS: Estimates coefficients of linear regression line using least squares
--> Use  number of points, means, cross deviation and x deviation to determine regression coefficients
'''
def est_coef(x,y):
    # find num of points
    n = np.size(x)

    # find mean of x and y
    meanX = np.mean(x)
    meanY = np.mean(y)

    # calc cross-deviation 
    cdXY = np.sum(y*x) - (n*meanY*meanX)

    # calc deviation about x
    dx = np.sum(x*x)-(n*meanX*meanX)

    # calc regression coefficients
    b1 = cdXY / dx
    b0 = meanY - (b1*meanX)
    return (b0,b1)


'''
INPUT: x (independent variable), y(dependant variable), b(coefficient tuple)
OUTPUT: scatter plot with linear regression line
PROCESS:
'''

def regressionLine(x,y,b):
    plt.figure(figsize=(10,4))
    # plot points as scatter plot
    plt.scatter(x,y,color="m",marker="o",s=30)

    # calc predicted response vector, ie predicted y values based on coefficient
    yPred = b[0]+(b[1]*x)

    # plot regression line
    plt.plot(x,yPred,color="g")

    # label plot
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

if __name__ == '__main__':
    x = np.array([0,1,2,3,4,5,6,7,8,9])
    y = np.array([1,3,2,5,7,8,8,9,10,12])

    # estimate coefficients
    b = est_coef(x,y)
    print(f"Estimated coefficients: {b[0]} and {b[1]}")
    regressionLine(x,y,b)