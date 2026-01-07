import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# import evaluation metric libraries for ML
import sklearn as sk
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, RepeatedStratifiedKFold

#import ML models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

#ignore warnings 
import warnings
warnings.filterwarnings("ignore")


iris = pd.read_csv('Iris_Classification/Iris.csv')
data = iris.iloc[:,1:]


def desc_sepal_length():
       #determine number & percentage of na values per column
    len1 = len(iris['SepalLengthCm'])
    c1 = iris['SepalLengthCm'].count()
    na1 = len1 - c1
    print('Number of missing values in SepalLengthCm: ', na1, '\nPercentage of missing values in SepalLengthCm: ', "{0:.1f}%".format(((float(na1)/len1)*100)))
        
    #determine mean, meadian, mode, standard deviation, min, max, and quartiles
    print('Min SepalLengthCm: ', iris['SepalLengthCm'].min(), '\nMax SepalLengthCm: ', iris['SepalLengthCm'].max(), '\nMean SepalLengthCm: ', 
          iris['SepalLengthCm'].mean(), '\nMedian SepalLengthCm: ', iris['SepalLengthCm'].median(), '\nMode SepalLengthCm: ', iris['SepalLengthCm'].mode(), 
          '\nStandard deviation SepalLengthCm: ', iris['SepalLengthCm'].std(), '\nQuartiles SepalLengthCm:\n ', iris['SepalLengthCm'].quantile([0.25, 0.5, 0.75]))
    
        
def desc_sepal_width():
       #determine number & percentage of na values per column
    len2 = len(iris['SepalWidthCm'])
    c2 = iris['SepalWidthCm'].count()
    na2 = len2 - c2
    print('Number of missing values in SepalWidthCm: ', na2, '\nPercentage of missing values in SepalWidthCm: ', "{0:.1f}%".format(((float(na2)/len2)*100)))
        
    #determine mean, meadian, mode, standard deviation, min, max, and quartiles
    print('Min SepalWidthCm: ', iris['SepalWidthCm'].min(), '\nMax SepalWidthCm: ', iris['SepalWidthCm'].max(), '\nMean SepalWidthCm: ', 
          iris['SepalWidthCm'].mean(), '\nMedian SepalWidthCm: ', iris['SepalWidthCm'].median(), '\nMode SepalWidthCm: ', iris['SepalWidthCm'].mode(), 
          '\nStandard deviation SepalWidthCm: ', iris['SepalWidthCm'].std(), '\nQuartiles SepalWidthCm:\n ', iris['SepalWidthCm'].quantile([0.25, 0.5, 0.75]))
        
def desc_petal_length():
       #determine number & percentage of na values per column
    len3 = len(iris['PetalLengthCm'])
    c3 = iris['PetalLengthCm'].count()
    na3 = len3 - c3
    print('Number of missing values in PetalLengthCm: ', na3, '\nPercentage of missing values in PetalLengthCm: ', "{0:.1f}%".format(((float(na3)/len3)*100)))
    
    #determine mean, meadian, mode, standard deviation, min, max, and quartiles
    print('Min PetalLengthCm: ', iris['PetalLengthCm'].min(), '\nMax PetalLengthCm: ', iris['PetalLengthCm'].max(), '\nMean PetalLengthCm: ', 
          iris['PetalLengthCm'].mean(), '\nMedian PetalLengthCm: ', iris['PetalLengthCm'].median(), '\nMode PetalLengthCm: ', iris['PetalLengthCm'].mode(), 
          '\nStandard deviation PetalLengthCm: ', iris['PetalLengthCm'].std(), '\nQuartiles PetalLengthCm:\n ', iris['PetalLengthCm'].quantile([0.25, 0.5, 0.75]))

        
def desc_petal_width():
       #determine number & percentage of na values per column
    len4 = len(iris['PetalWidthCm'])
    c4 = iris['PetalWidthCm'].count()
    na4 = len4 - c4
    print('Number of missing values in PetalWidthCm: ', na4, '\nPercentage of missing values in PetalWidthCm: ', "{0:.1f}%".format(((float(na4)/len4)*100)))
    # determine mean, meadian, mode, standard deviation, min, max, and quartiles
    print('Min PetalWidthCm: ', iris['PetalWidthCm'].min(), '\nMax PetalWidthCm: ', iris['PetalWidthCm'].max(), '\nMean PetalWidthCm: ', 
          iris['PetalWidthCm'].mean(), '\nMedian PetalWidthCm: ', iris['PetalWidthCm'].median(), '\nMode PetalWidthCm: ', iris['PetalWidthCm'].mode(),
          '\nStandard deviation PetalWidthCm: ', iris['PetalWidthCm'].std(), '\nQuartiles PetalWidthCm:\n ', iris['PetalWidthCm'].quantile([0.25, 0.5, 0.75]))

#determine descriptive statistics
def descriptive_stats():
    #get all columns:
    cols = iris.columns
    print(cols)
    
    print('Determining descriptive statistics of numerical values of sepal length')
    desc_sepal_length()
    print('-----------------------')
    
    print('Determining descriptive statistics of numerical values of sepal width')
    desc_sepal_width()
    print('-----------------------')
    
    print('Determining descriptive statistics of numerical values of petal length')
    desc_petal_length()
    print('-----------------------')
    
    print('Determining descriptive statistics of numerical values of petal width')
    desc_petal_width()
    print('-----------------------')
    

def plots_scatter():
    #create multi scatter plots of numerical columns and color code by species
   g = sns.PairGrid(iris)
   g.map(sns.scatterplot, hue=iris['Species'])
   g.add_legend()
   plt.show() 
    
    
def plots_multi():
    #create multi plots (various) of numerical columns and color code by species where applicable
    g = sns.PairGrid(iris)
    g.map_upper(sns.scatterplot, hue=iris['Species'])
    g.map_lower(sns.kdeplot)
    g.map_diag(sns.kdeplot, lw=3, legend=False)
    plt.show()

def correlation():
    #create correlation heatmap
    cors = iris[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']].corr()
    plt.figure(figsize=(10,4))
    sns.heatmap(cors, annot=True, cmap='Reds_r')
    plt.title('Correlation Matrix')
    plt.show()
  
'''
INPUT: The model to be evaluated, the training data and the test data
MODEL OPTIONS: [Logistic Regression, Decision Tree, Random Forest, SVM, Xtreme Gradient Boosting, Naive Bayes, Neural Network]
OUTPUT: Returns model scores as lists - recall_train, recall_test, precision_train, precision_test, f1_train, f1_test

PROCESS:
- Fits the given model using training data
- Makes predictions on trained model
- Find ROC_AUC score of train and test data & plots ROC curve and AUC curve
- Plots confusion matrix for train and test data
- Prints classification report for train and test data
- Plots important features if they exist
- Returns model scores as lists - recall_train, recall_test, precision_train, precision_test, f1_train, f1_test


'''
def model_eval(model, trainX, trainY, testX, testY):
    # fit model to training data
    model.fit(trainX, trainY)
    
    # make predicitons on test data
    yPredictTrain = model.predict(xTrain)
    yPredictTest = model.predict(xTest)
 
    #calculate confusion matrix
    cmTrain = confusion_matrix(yTrain, yPredictTrain)
    cmTest = confusion_matrix(yTest, yPredictTest)
    
    #plot confusion matrix
    #    sns.heatmap(cm_train, annot=True, xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'], cmap="Oranges", fmt='.4g', ax=ax[0])

    fig, ax = plt.subplots(1,2,figsize=(11,4))
    sns.heatmap(cmTrain, annot=True, xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'], cmap="Oranges", fmt='.4g', ax=ax[0])
    ax[0].set_title('Confusion Matrix Train')
    ax[0].set_ylabel('True Label')
    ax[0].set_xlabel('Predicted Label')

    sns.heatmap(cmTest, annot=True, xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'], cmap="Oranges", fmt='.4g', ax=ax[1])
    ax[1].set_title('Confusion Matrix Test')
    ax[1].set_ylabel('True Label')
    ax[1].set_xlabel('Predicted Label')

    plt.tight_layout()
    plt.show()
    
    #classification report
    crTrain = classification_report(yTrain, yPredictTrain, output_dict=True)
    crTest = classification_report(yTest, yPredictTest, output_dict=True)
    print("\n Classification Report Train")
    crT = pd.DataFrame(crTrain).T
    print(crT.to_markdown())
    print("\n Classification Report Test")
    crT = pd.DataFrame(crTest).T
    print(crT.to_markdown())
    
    #calculate model scores
    precisionTrain = crTrain['weighted avg']['precision']
    precisionTest = crTest['weighted avg']['precision']
    
    recallTrain = crTrain['weighted avg']['recall']
    recallTest = crTest['weighted avg']['recall']

    accTrain = accuracy_score(y_true = yTrain, y_pred=yPredictTrain)
    accTest = accuracy_score(y_true = yTest, y_pred=yPredictTest)
    
    f1Train = crTrain['weighted avg']['f1-score']
    f1Test = crTest['weighted avg']['f1-score']
    
    
    
    modelScores = [precisionTrain, precisionTest, recallTrain, recallTest, accTrain, accTest, f1Train, f1Test]
    return modelScores

'''
    INPUT: The model we're tuning
    OUTPUT: The optimal parameters for the model
    
    PROCESS:
    Uses GridSearchCV to find optimal parameters using exhaustive search on a small parameter space.
'''
def hp(model, xTrain, yTrain):
    if(model == "Logistic Regression"):
        paramGrid = {
                    'C':[100,10,1,0.1,0.01,0.001,0.0001],
                    'penalty':['l1','l2'],
                    'solver':['newton-cg','lbfgs','liblinear','sag','saga']
                   }
        #initialize model
        lr = LogisticRegression(fit_intercept=True, max_iter=10000, random_state=0)
        # repeated stratified kfold cross validation
        rskf = RepeatedStratifiedKFold(n_splits=3, n_repeats=4, random_state=0)
        # grid search cross validation
        grid = GridSearchCV(lr, paramGrid, cv=rskf)
        grid.fit(xTrain, yTrain)
        params = grid.best_params_
    elif(model == "Decision Tree"):
        grid = {
            'max_depth':[3,4,5,6,7,8],
            'min_samples_leaf':np.arange(2,8),
            'min_samples_split':np.arange(10,20)
        }
        #initialize model
        m = DecisionTreeClassifier()
        # repeated stratified kfold cross validation
        rskf = RepeatedStratifiedKFold(n_splits=3, n_repeats=3, random_state=0)
        #initialize grid search cross validation
        grid_search = GridSearchCV(m, grid, cv=rskf)
        grid_search.fit(xTrain, yTrain)
        params = grid_search.best_params_
        
    elif(model=="Random Forest"):
        grid={'n_estimators':[10,50,100,200],
              'max_depth':[8,9,10,11,12,13,14,15],
              'min_samples_split':[2,3,4,5] }  
        #initialize model
        rf = RandomForestClassifier(random_state=0)
        # repeated stratified kfold cross validation
        rskf = RepeatedStratifiedKFold(n_splits=3,n_repeats=3,random_state=0)
        #initialize randomsearchcv
        randomSearch = RandomizedSearchCV(rf,grid,cv=rskf,n_iter=10,n_jobs=-1)
        #fit randomSearch to training data
        randomSearch.fit(xTrain,yTrain)
        params = randomSearch.best_params_
    
    return params
    
if __name__ == '__main__':
    
   # descriptive_stats()

    #plots_scatter()
    
   # plots_multi()
    
   # correlation()
   
   # encode species as numerical values
    labelEncoder = LabelEncoder()
    
    data["Species"] = labelEncoder.fit_transform(data["Species"])
    #get unique values
    unique = data["Species"].unique()
    print("Encoded Species values are: ", unique)
    
    # drop orig categorical column species
    x = data.drop('Species', axis=1)
    y = data['Species']
    
    # split date into train xy and test xy where x is independent and y is dependent
    xTrain, xTest, yTrain, yTest = train_test_split(x,y,test_size=0.3)
    
    # check distribution of dependent variable (y) for each species
    yTrain.value_counts()
    
    # create model score dataframe
    score = pd.DataFrame(index=["Precision Train", "Precision Test", "Recall Train", "Recall Test", "Accuracy Train", "Accuracy Test", "F1 Train", "F1 Test"])
    
    print("-----Model Evaluation----")
    print("Logistic Regression")
    '''-------Logistic Regression-----------'''
    #create logisitic regression model
    lr = LogisticRegression(fit_intercept=True, max_iter=10000)
    lrScore = model_eval(lr, xTrain, yTrain, xTest, yTest)
    
    #update model scores with lr score
    score["Logistic Regression"] = lrScore
    # perform hyperparameter tuning & cross validation
    params = hp("Logistic Regression", xTrain, yTrain)
    
    # initialize model with optimal parameters
    print('Logistic Regression Tuned')
    lr = LogisticRegression(C=params['C'], penalty=params['penalty'], solver=params['solver'], max_iter=10000, random_state=0)
    lrScore = model_eval(lr, xTrain, yTrain, xTest, yTest)
    
    #update model scores with lr score
    score["Logistic Regression tuned"] = lrScore
    print(score)
    
    '''-------Decision Tree-----------'''
    print("\nDecision Tree")
    dt = DecisionTreeClassifier(random_state=20)
    dtScore = model_eval(dt, xTrain, yTrain, xTest, yTest)
    
    #update model scores with dt score
    score["Decision Tree"] = dtScore
    
    params = hp("Decision Tree", xTrain, yTrain)
    
    #initialize model with optimal parameters
    dt = DecisionTreeClassifier(max_depth=params['max_depth'], 
                                min_samples_leaf=params['min_samples_leaf'],
                                min_samples_split=params['min_samples_split'],
                                random_state=20)
    dtScore = model_eval(dt, xTrain, yTrain, xTest, yTest)
    
    #update model scores with dt score
    score["Decision Tree tuned"] = dtScore
    print(score)
    
    '''-------Random Forest-----------'''
    print("\n Random Forest")
    rf = RandomForestClassifier(random_state=0)
    rfScore = model_eval(rf,xTrain,yTrain,xTest,yTest)
    #update model scores with rf score
    score["Random Forest"] = rfScore
    params = hp("Random Forest",xTrain, yTrain)

    #initialize model with optimal parameters
    rf=RandomForestClassifier(n_estimators=params['n_estimators'],
                              min_samples_leaf=params['min_samples_split'],
                              max_depth=params['max_depth'],
                              random_state=0)
    rfScore=model_eval(rf,xTrain,yTrain,xTest,yTest)
    #update model score
    score["Random Forest tuned"] = rfScore
    print(score)