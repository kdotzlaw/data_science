'''
A predictive model that can differentiate between bad cnxns, called intrusions/attacks
and good cnxns
Attack Categories: 
- DOS/DDOS: denial of service ex) syn flooding
- R2L: unauthorized access from remote machine
- U2L: unauthorized access to local superuser privileges ex) buffer overflow
- probing: ex) port scanning
'''
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


'''
---eda---
INPUT: dataframe
OUTPUT: split data into trainX, testX, trainY, testY
PROCESS:
- find categorical features
- visualize categorical features
- preprocess data by filtering numerical cols & prepping feature matrix
- split data into train/test split
- feature encoding: protocol types->ints, flags->ints
'''
def eda(df):
    # Find categorical features
    nums = df._get_numeric_data().columns
    categories = list(set(df.columns)-set(nums))

    # remove target & attack type
    categories.remove('target')
    categories.remove('Attack Type')
    #print(categories)

    # create bar graphs of categorical features
    df['protocol_type'].value_counts().plot(kind='bar')
    plt.show()
    df['logged_in'].value_counts().plot(kind='bar')
    plt.show()
    df['Attack Type'].value_counts().plot(kind='bar')
    plt.show()
    
    # drop target column & empty
    df = df.drop(['target'],axis=1)
    df = df.dropna(axis='columns')

    # filter numeric columns
    ndf = df[[col for col in df.columns if df[col].nunique() > 1 and pd.api.types.is_numeric_dtype(df[col])]]

    # feature matrix X and target variable Y
    y = df[['Attack Type']]
    x = df.drop(['Attack Type'],axis=1)

    # split data into train/test -- 67/33
    trainX, testX, trainY, testY = train_test_split(x, y, test_size=0.33,random_state=42)

    # map protocol_type to ints & update train & test data
    protomap = {'icmp': 0, 'tcp':1,'udp':2}
    trainX['protocol_type'] = trainX['protocol_type'].map(protomap)
    testX['protocol_type'] = testX['protocol_type'].map(protomap)

    # map flag to ints & update train & test data
    flagmap = {'SF':0, 'S0':1,'REJ':2,'RSTR':3,'RSTO':4,'SH':5,'S1':6,'S2':7,'RSTOS0':8,'S3':9,'OTH':10}
    trainX['flag'] = trainX['flag'].map(flagmap)
    testX['flag'] = testX['flag'].map(flagmap)

    return trainX, trainY, testX, testY


'''
---correlation---
INPUT: trainX, trainY, testX, testY
OUTPUT: correlation heatmap, heatmap without highly correlated features
PROCESS:
- numeric feature correlation heatmap with trainX
- remove highly correlated features & create heatmap
- drop columns that dont add value
- create correlation heatmap new dataset
'''
def correlation(trainX, trainY, testX, testY):
    # get numeric features for heatmap
    ntrainX = trainX.select_dtypes(include=['float64','int64'])
    corr = ntrainX.corr()

    # corr heatmap
    plt.figure(figsize=(12,10))
    sns.heatmap(corr,cmap='coolwarm',linewidths=0.5)
    plt.title('Feature Correlation Heatmap on Training Set')
    plt.tight_layout()
    plt.show()

    # remove highly correlated features
    hc = ['num_root','srv_serror_rate',
          'dst_host_srv_serror_rate','dst_host_rerror_rate',
          'dst_host_srv_rerror_rate','dst_host_same_srv_rate']
    trainX.drop(columns=hc, axis=1,inplace=True)
    testX.drop(columns=hc, axis=1,inplace=True)

    # remove cols that dont add value
    trainX.drop(['is_host_login', 'num_outbound_cmds'], axis=1, inplace=True)
    testX.drop(['is_host_login', 'num_outbound_cmds'], axis=1, inplace=True)
    trainX.drop('service', axis=1, inplace=True)
    testX.drop('service', axis=1, inplace=True)

    # corr heatmap on transformed data
    ntrainX = trainX.select_dtypes(include=['float64','int64'])
    corr = ntrainX.corr()

    plt.figure(figsize=(12,10))
    sns.heatmap(corr,cmap='coolwarm',linewidths=0.5)
    plt.title('Feature Correlation Heatmap on Training Set')
    plt.tight_layout()
    plt.show()

'''
---trainModel---
INPUT: trainX, trainY, testX, testY
OUTPUT: 
PROCESS:

'''
def trainModel(trainX, trainY, testX, testY):
    # init models
    models = {
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(criterion="entropy", max_depth=4),
        "Random Forest": RandomForestClassifier(n_estimators=30),
        "SVM": LinearSVC(),
        "Logistic Regression": LogisticRegression(max_iter=1200000),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=30, max_depth=4, random_state=0),
    }

    trainScores = []
    testScores = []
    trainTime = []
    testTime = []

    # Train models
    for name, model in models.items():
        print(f"Training {name}...")

        # train model
        start = time.time()
        model.fit(trainX, trainY.values.ravel())
        end = time.time()
        train_time = end-start
        trainTime.append(train_time)

        # predictions
        start = time.time()
        predTrain = model.predict(trainX)
        predTest  = model.predict(testX)
        end = time.time()
        test_time = end-start
        testTime.append(test_time)

        # calculate train & test accuracy scores
        trainScore = accuracy_score(trainY, predTrain)*100
        trainScores.append(trainScore)
        testScore = accuracy_score(testY,predTest)*100
        testScores.append(testScore)

        print(f"{name} \n Train Accuracy: {trainScore:.2f}% \n Test Accuracy: {testScore:.2f}%")
        print(f"Training Time: {train_time:.4f}s, Test Time: {test_time:.4f}s")

if __name__ == "__main__":
    # read feature list
    with open('kddcup.names.txt','r') as f:
        print(f.read())

    # append read data columns to dataset
    cols = """duration,
            protocol_type,
            service,
            flag,
            src_bytes,
            dst_bytes,
            land,
            wrong_fragment,
            urgent,
            hot,
            num_failed_logins,
            logged_in,
            num_compromised,
            root_shell,
            su_attempted,
            num_root,
            num_file_creations,
            num_shells,
            num_access_files,
            num_outbound_cmds,
            is_host_login,
            is_guest_login,
            count,
            srv_count,
            serror_rate,
            srv_serror_rate,
            rerror_rate,
            srv_rerror_rate,
            same_srv_rate,
            diff_srv_rate,
            srv_diff_host_rate,
            dst_host_count,
            dst_host_srv_count,
            dst_host_same_srv_rate,
            dst_host_diff_srv_rate,
            dst_host_same_src_port_rate,
            dst_host_srv_diff_host_rate,
            dst_host_serror_rate,
            dst_host_srv_serror_rate,
            dst_host_rerror_rate,
            dst_host_srv_rerror_rate"""
    columns = []
    for c in cols.split(',\n'):
        if c.strip():
            columns.append(c.strip())
    # add target column
    columns.append('target')
    print(len(columns))

    # read attack_types
    with open("training_attack_types.txt",'r') as f:
        print(f.read())

    # create dict of attack types & categories
    attackTypes = {
        'normal': 'normal',
        'back': 'dos',
        'buffer_overflow': 'u2r',
        'ftp_write': 'r2l',
        'guess_passwd': 'r2l',
        'imap': 'r2l',
        'ipsweep': 'probe',
        'land': 'dos',
        'loadmodule': 'u2r',
        'multihop': 'r2l',
        'neptune': 'dos',
        'nmap': 'probe',
        'perl': 'u2r',
        'phf': 'r2l',
        'pod': 'dos',
        'portsweep': 'probe',
        'rootkit': 'u2r',
        'satan': 'probe',
        'smurf': 'dos',
        'spy': 'r2l',
        'teardrop': 'dos',
        'warezclient': 'r2l',
        'warezmaster': 'r2l',
    }

    # read into dataframe
    df = pd.read_csv("kddcup.data_10_percent_corrected",names=columns)
    # add attackType column
    df['Attack Type'] = df.target.apply(lambda r:attackTypes[r[:-1]])

    # data exploration
    trainX, trainY, testX, testY = eda(df)
    correlation(trainX, trainY, testX,testY)

    # scale trainX & testX
    sc = MinMaxScaler()
    trainX = sc.fit_transform(trainX)
    testX = sc.fit_transform(testX)

    trainModel(trainX, trainY, testX, testY)