import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")


def survival_trends(train):
    # survivors and death count visuals
    fig, ax = plt.subplots(1,2,figsize=(12,4))

    train['Survived'].value_counts().plot.pie(
        explode=[0,0.1], 
        autopct='%1.1f%%',
        ax=ax[0],
        shadow=False
    )

    ax[0].set_title('Survivors 1, dead 0')
    ax[0].set_ylabel('')
    sns.countplot(x='Survived',data = train, ax=ax[1])
    ax[1].set_ylabel('Quantity')
    ax[1].set_title('Survivors 1, dead 0')
    plt.show()

    # survival by sex visuals
    fig, ax = plt.subplots(1,2,figsize=(12,4))
    train[['Sex','Survived']].groupby(['Sex']).mean().plot.bar(ax=ax[0])
    ax[0].set_title('Survivors by Sex')
    sns.countplot(x='Sex',hue='Survived',data=train,ax=ax[1])
    ax[1].set_ylabel('Quantity')
    ax[1].set_title('Survived 1, deceased 0: men and women')
    plt.show()
    
'''
INPUT: the training dataset
OUPUT: accuracy score of model
PROCESS:
Tracks predictors and target, splits data into x and y training data, 
fits data to random forest model, tracks prediction results, determines model accuracy

'''

def trainModel(train):
    # drop predictors from training data (survived & passenger id)
    predictors = train.drop(['Survived','PassengerId'],axis=1)

    # determine what is to be predicted
    target = train['Survived']

    # split data
    trainX, xVal, trainY, yVal = train_test_split(predictors,target,test_size=0.2,random_state=0)

    # init model - random forest
    rf = RandomForestClassifier()

    # fit training data
    rf.fit(trainX,trainY)

    # track predictions
    yPred = rf.predict(xVal)

    # Predict survival based on passenger id
    ids = test['PassengerId']
    predictions = rf.predict(test.drop('PassengerId',axis=1))
    output = pd.DataFrame({"PassengerId":ids,'Survived':predictions})
    output.to_csv('results.csv',index=False)

    return round(accuracy_score(yPred,yVal)*100,2)



if __name__=="__main__":
    plt.style.use('fivethirtyeight')

    # read data into dfs
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')

    # get basic info
    print(train.shape)
    print(train.info())

    # determine null values in dataset
    print(train.isnull().sum())

    # create plots for survivors and death counts from training data
    # and how sex impacted survival
    survival_trends(train)
    print("----Feature Engineering----")
    '''feature engineering 
        - drop redundant features (ie features that have little predictive value)
        - create new columns 
        - convert text into numerical data
    '''
    # drop cabin, ticket
    train = train.drop(['Cabin'],axis=1)
    test = test.drop(['Cabin'],axis=1)
    train = train.drop(['Ticket'],axis=1)
    test = test.drop(['Ticket'],axis=1)

    # populate Null values in Embarked
    train = train.fillna({'Embarked':'S'})

    # sort age into groups (ie convert into categorical data)
    train['Age'] = train['Age'].fillna(-0.5)
    test['Age'] = test['Age'].fillna(-0.5)
    # create numerical values to assign to groups
    bins=[-1,0,5,12,18,24,35,60,np.inf]
    labels=['Unknown','Baby','Child','Teen','Student','Young Adult','Adult','Senior']
    train['AgeGroup'] = pd.cut(train['Age'],bins,labels=labels)
    test['AgeGroup']=pd.cut(test['Age'],bins,labels=labels)

    # combine datasets & extract title for each name in dataset
    comb = [train,test]
    for dataset in comb:
        dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.',expand=False)
    pd.crosstab(train['Title'],train['Sex'])
    # replace titles with common ones
    for dataset in comb:
        dataset['Title'] = dataset['Title'].replace(
            ['Lady', 'Capt', 'Col',
              'Don', 'Dr', 'Major',
              'Rev', 'Jonkheer', 'Dona'],
            'Rare')
        dataset['Title'] = dataset['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
        dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    train[['Title','Survived']].groupby(['Title'], as_index=False).mean()
    
    # map to numerical values
    tmap = {'Mr':1,"Miss":2,"Mrs":3,"Master":4,"Royal":5,"Rare":6}
    for dataset in comb:
        dataset['Title']=dataset['Title'].map(tmap)
        dataset['Title']=dataset['Title'].fillna(0)
    
    # fill in missing age values using the mode (most freq occuring)
    mrAge = train[train["Title"] == 1]["AgeGroup"].mode() #sb young adult
    missAge =train[train["Title"] == 2]["AgeGroup"].mode() #sb student
    mrsAge = train[train["Title"] == 3]["AgeGroup"].mode() #sb adult
    mAge = train[train["Title"] == 4]["AgeGroup"].mode() #sb baby
    royAge = train[train["Title"] == 5]["AgeGroup"].mode() #sb adult
    raAge = train[train["Title"] == 6]["AgeGroup"].mode() #sb adult

    # Map unknowns to title map
    ageTitleMap = {1:"Young Adult",2:"Student",3:"Adult",4:"Baby",5:"Adult",6:"Adult"}
    for i in range(len(train['AgeGroup'])):
        if train["AgeGroup"][i] == "Unknown":
            train['AgeGroup'][i] = ageTitleMap[train['Title'][i]]

    for i in range(len(test["AgeGroup"])):
        if test['AgeGroup'][i]=="Unknown":
            test['AgeGroup'][i] = ageTitleMap[test['Title'][i]]

    # assign numerical values to age categories
    ageMap = {'Baby': 1, 'Child': 2, 'Teenager': 3,
               'Student': 4, 'Young Adult': 5, 'Adult': 6, 
               'Senior': 7} 
    train["AgeGroup"] = train["AgeGroup"].map(ageMap)
    test['AgeGroup'] = test['AgeGroup'].map(ageMap)

    # drop age feature since AgeGroup exists
    train = train.drop(["Age"],axis=1)
    test = test.drop(["Age"],axis=1)

    # drop name bc it wont help make useful predictions
    train = train.drop(["Name"],axis=1)
    test = test.drop(["Name"],axis=1)

    # assign numerical values to Sex
    sMap = {"Male":0,"Female":1}
    train['Sex'] = train['Sex'].map(sMap)
    test['Sex'] = test['Sex'].map(sMap)

    # assign numerical values to Embarks
    eMap = {"S":1,"C":2,"Q":3}
    train['Embarked'] = train['Embarked'].map(eMap)
    test['Embarked'] = test['Embarked'].map(eMap)

    # Fill in missing values in Fare based on mean fare for that PClass
    for i in range(len(test['Fare'])):
        if pd.isnull(test['Fare'][i]):
            pclass = test['Pclass'][i]
            test['Fare'][i] = round(train[train['Pclass'] == pclass]['Fare'].mean(),4)

    # Map into quartile groups & drop Fare values
    train['FareBand']=pd.qcut(train['Fare'],4,labels=[1,2,3,4])
    test['FareBand']=pd.qcut(test['Fare'],4,labels=[1,2,3,4])
    train = train.drop(['Fare'],axis=1)
    test=test.drop(['Fare'],axis=1)
    print("----Training Model----")
    accuracy = trainModel(train)
    print(f"Accuracy Score: {accuracy}")
