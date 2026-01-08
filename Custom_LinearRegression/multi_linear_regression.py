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
    x = np.hstack([rdf.values[::2,:]],
                  rdf.values[1::2,:2])
    y = rdf.values[1::2,2]

    # split data into training and test data