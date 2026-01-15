'''
Predict development of heart disease using Logistic Regression, Decision Trees, and Random Forest
'''
# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

if __name__=='__main__':
    # csv to dataframe
    df = pd.read_csv('dataset.csv')
   
    # drop education, doesnt effect hd development
    df.drop('education',axis=1,inplace=True)
    # rename TenYearCHD to CHD
    df.rename(columns={'TenYearCHD':'CHD'},inplace=True)
    print(df.head())

    # train-test split 80/20
    trainX,trainY,testX,testY = train_test_split(df.iloc[:,:-1],df.iloc[:,-1],test_size=0.2)
    # concat x,y train & test data
    train = pd.concat([trainX,trainY],axis=1)
    test = pd.concat([testX,testY],axis=1)
    print(train.head())
    # plot male CHD instances
    sns.countplot(x=train['male'],hue=train['CHD'])

    # correlation heatmap of df
    plt.figure(figsize=(15,15))
    sns.heatmap(train.corr(),annot=True,linewidths=0.1)

