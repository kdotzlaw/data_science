import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler
import warnings
warnings.filterwarnings('ignore')

'''
INPUT: the dataframe
OUTPUT: the same dataframe but with new column
PROCESS:
- create SumScores column with sum of all A scores


'''
def addFeature(df):
    # create new column with 0 values
    df['sumScores'] = 0
    for col in df.loc[:,'A1_Score':'A10_Score'].columns:
        df['sumScores']+=df[col]
    df['ind']=df['autism']+df['used_app_before']+df['jaundice']
    return df
   
'''
INPUT: numerical age
OUTPUT: ageGroup column in dataframe
PROCESS: Assign categorical label to ages

'''
def convAge(age):
    if age<4:
        return 'Toddler'
    elif age<12:
        return 'Kid'
    elif age<18:
        return 'Teenager'
    elif age<65:
        return 'Adult'
    else:
        return 'Senior'

'''
INPUT: dataframe
OUTPUT: dataframe with encoded labels
PROCESS:
- check if column is object
    - if yes, encode it with LabelEncoder & fit_transform it
'''
def encode(df):
    for col in df.columns:
        if df[col].dtype=='object':
            le = LabelEncoder()
            df[col]=le.fit_transform(df[col])
    return df

'''
EDA of Autism Data
INPUT: dataframe
OUTPUT:
PROCESS:

'''
def eda(df):
    # create chart of percent of ASD
    plt.pie(df['Class/ASD'].value_counts().values,autopct='%1.1f%%',labels=['Not Autistic','Autistic'])
    plt.show()

    # since data is imbalanced, seperate columns based on data type
    ints = []
    objs = []
    floats =  []
    for col in df.columns:
        if df[col].dtype==int:
            ints.append(col)
        elif df[col].dtype==object:
            objs.append(col)
        else:
            floats.append(col)

    # remove ID (not helpful) and Class/ASD (already looked at) columns
    ints.remove('ID')
    ints.remove('Class/ASD')

    # convert data to long form with melt
    df_melt = df.melt(id_vars=['ID','Class/ASD'],value_vars=ints,var_name='col',value_name='value')
    '''
    # create subplots for ints
    fig,axes = plt.subplots(5,3,figsize=(25,15))
    axes = axes.flatten()
    for i, col in enumerate(ints):
        # create a plot with legend, then share it with the rest
        plt.subplot(5,3,i+1)
        if i==0:
            sb.countplot(x='value',hue='Class/ASD', data = df_melt[df_melt['col']==col])
            handles, labels = axes[0].get_legend_handles_labels()
            axes[i].get_legend().remove()
        else:
            sb.countplot(x='value',hue='Class/ASD', data = df_melt[df_melt['col']==col],legend=False)
        axes[i].set_xlabel(col)
    fig.legend(handles,labels,title='Class/ASD',loc='upper right',bbox_to_anchor=(0.98,0.98))
    plt.subplots_adjust(hspace=0.6) # add padding so labels show
    plt.show()
    

    # create subplots for objects
    plt.subplots(figsize=(15,15))
    for i, col in enumerate(objs):
        plt.subplot(5,3,i+1)
        sb.countplot(x=col,hue='Class/ASD',data=df)
        plt.title(f"Distribution of {col}")
        plt.xticks(rotation=45,ha='right')
    plt.tight_layout()
    plt.show()

    # create single plot for country_of_res because its squished
    plt.figure(figsize=(15,5))
    sb.countplot(data=df,x='country_of_res',hue='Class/ASD')
    plt.xticks(rotation=90)
    plt.show()
    
    # create distribution plot for float data
    plt.subplots(figsize=(15,5))
    for i,col in enumerate(floats):
        plt.subplot(1,2,i+1)
        sb.histplot(df[col])
    plt.tight_layout()
    plt.show()

    # create box plot of float data to view data skew
    plt.subplots(figsize=(15,5))
    for i,col in enumerate(floats):
        plt.subplot(1,2,i+1)
        sb.boxplot(df[col])
    plt.tight_layout()
    plt.show()
    '''
    # remove outliers in result column
    df = df[df['result']>-5]
    
    # Feature Engineering - convert Ages to categories
    df['ageGroup']=df['age'].apply(convAge)
    sb.countplot(x=df['ageGroup'],hue=df['Class/ASD'])
    plt.show()

    # Feature Engineering - convert A scores into a score summary
    addFeature(df)
    sb.countplot(x=df['sumScores'],hue=df['Class/ASD'])
    plt.show()

    # Remove data skew with Log Transformations
    df['age']=df['age'].apply(lambda x:np.log(x))
    sb.displot(df['age'])
    plt.show()

    
    

if __name__=="__main__":

    # read in csv into data frame
    df = pd.read_csv('train.csv')

    print("Data basics....")
    print(f"Shape: {df.shape}")
    print("Dataset info: ")
    print(df.info())
    print("Description of data in each column: ")
    print(df.describe().T)
   
    # Data cleaning
    print("Data cleaning...")
    print(f"Ethnicity column: {df['ethnicity'].value_counts()}")
    print(f"Relation column: {df['relation'].value_counts()}")
    df = df.replace({'yes':1,'no':0,'?':'others','others':'others'})

    # Explore dataset
    eda(df)

    # Encode object labels
    df = encode(df)

    # Create heatmap of correlations where >0.8 is highly correlated
    plt.figure(figsize=(10,10))
    sb.heatmap(df.corr()>0.8,annot=True,cbar=False)
    plt.show()

    # Seperate Features & Target variables
    rm = ['ID','age_desc','used_app_before','autism']
    features = df.drop(rm+['Class/ASD'],axis=1)
    target = df['Class/ASD']

    # Split into training and testing (80/20)
    trainX, testX, trainY, testY = train_test_split(features, target,test_size=0.2,random_state=10)

    # Balance data by adding repeat minority rows with RandomOverSampler
    ros = RandomOverSampler(sampling_strategy='minority',random_state=0)
    X,Y = ros.fit_resample(trainX,trainY)
    print(f"{X.shape},{Y.shape}")

    # Normalize data with StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    testX = scaler.fit_transform(testX)

    # Train Linear Regression, XGBClassifier, SVC Models
    models = [LogisticRegression(),XGBClassifier(),SVC(kernel='rbf')]
    for model in models:
        model.fit(X,Y)
        print(f"{model}")
        print(f"Training Accuracy: {metrics.roc_auc_score(Y,model.predict(X))}")
        print(f"Validation Accuracy: {metrics.roc_auc_score(testY,model.predict(testX))}\n")