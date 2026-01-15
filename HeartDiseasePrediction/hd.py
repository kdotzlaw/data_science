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


    x = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    trainX, testX, trainY, testY = train_test_split(x, y, test_size=0.20)
    
    train = pd.concat([trainX, trainY], axis=1)
    test = pd.concat([testX, testY], axis=1)

    ax = sns.countplot(x=train['male'], hue=train['CHD']) 
    # change tick labels 0.0 and 1.0 for males
    ax.set_xticklabels(['No','Yes'])
    # change legend labels 0.0 and 1.0 for CHD
    for text in ax.legend_.get_texts():
        if text.get_text()=='0.0':
            text.set_text("No")
        elif text.get_text()=='1.0':
            text.set_text("Yes")
    plt.show()
    
    # correlation heatmap of df
    plt.figure(figsize=(15,15))
    sns.heatmap(train.corr(),annot=True,linewidths=0.1)
    plt.show()