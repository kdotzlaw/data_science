'''
Predict development of heart disease using Logistic Regression, Decision Trees, and Random Forest
'''
# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import  DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
'''

'''
def createPlots(train):
    
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

'''
INPUT: training data xy, test data xy
OUTPUT: model accuracy score
PROCESS: fit model with training data, predict target using testX, calculate accuracy
'''
def logisticRegression(trainX,trainY,testX,testY):
    # init LR classifier
    lr = LogisticRegression() 
    
    # fit lr with training data
    fitted = lr.fit(trainX,trainY)

    # predict value using testX
    pred = fitted.predict(testX)

    return accuracy_score(pred,testY)*100
'''
INPUT: training data xy, test data xy
OUTPUT: accuracy score of decision tree
PROCESS:
- use tree with max_depth 3
- fit tree to training data
- predict target using testX
- calculate accuracy
'''
def decisionTree(trainX,trainY,testX,testY):
    tree = DecisionTreeClassifier(max_depth=3)
    # fit test data
    fitted = tree.fit(trainX,trainY)
    # predict target
    pred = fitted.predict(testX)
    return accuracy_score(pred,testY)*100

'''
INPUT: training data xy, test data xy
OUTPUT: accuracy score of random forest
PROCESS:
- use forest with estimator 3
- fit tree to training data
- predict target using testX
- calculate accuracy
'''
def randomForest(trainX,trainY,testX,testY):
    rf = RandomForestClassifier(n_estimators=3)
    # fit with k nearest neighbour
    knn = rf.fit(trainX,trainY)
    # predict target
    pred = knn.predict(testX)
    return accuracy_score(pred,testY)*100

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

    # create sex<->CHD plot & heatmap
    createPlots(train)

    # drop highly correlated columns
    train.drop(['currentSmoker','diaBP'],axis=1,inplace=True)
    # remove outliers in training data to reduce inflation
    train = train[~(train['sysBP']>220)]
    train = train[~(train['BMI']>43)]
    train = train[~(train['heartRate']>125)]
    train = train[~(train['glucose']>200)]
    train = train[~(train['totChol']>450)]

    #print(train.keys())
    # standardize columns into array
    cols = ['age','cigsPerDay','totChol','sysBP','BMI','heartRate','glucose']
    #print(f"Columns in train:{train.columns.tolist()}")
    #missing = [col for col in cols if col not in train.columns]
    #print(f"Missing columns: {missing}")
    scaler = StandardScaler()
    scaledTrain=scaler.fit_transform(train[cols]) #np array
    train = pd.DataFrame(scaledTrain,columns=cols,index=train.index)
    
    # fit data
    test.drop(['currentSmoker','diaBP'],axis=1,inplace=True)

    # fill null values
    imp = SimpleImputer(strategy='most_frequent')
    test = pd.DataFrame(imp.fit_transform(test))
   
    # Find Accuracy of Logistic Regression
    lr = round(logisticRegression(trainX,trainY,testX,testY),2)
    print(f"Logistic Regression {lr}")
    # Find Accuracy of Decision Tree
    dt = round(decisionTree(trainX,trainY,testX,testY),2)
    print(f"Decision Tree {dt}")
    
    rf = round(randomForest(trainX,trainY,testX,testY),2)
    print(f"Random Forest {rf}")
    accuracy = [lr,dt,rf]
    m = max(accuracy)
    if accuracy[0] == m:
        print(f"Logistic Regression is best model with accuracy {m}")
    elif accuracy[1]==m:
        print(f"Decision Tree is best model with accuracy {m}")
    else:
        print(f"Random Forest is best model with accuracy {m}")