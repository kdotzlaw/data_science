import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

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
    
    # assign numerical values
    
    # fill in missing age values 
