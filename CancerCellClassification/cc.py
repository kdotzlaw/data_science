import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    # load data into dataframe
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)

    # EDA
    print(df.info())
    print("-----------------")
    print(df.sample(5))
    print("-----------------")
    print(df.describe())
    print("-----------------")
    dff = pd.DataFrame(data.target, columns=['target'])

    # create distribution chart -- 0 malignant, 1 benign
    cc = dff['target'].value_counts()
    plt.pie(cc,labels=['1: Benign','0: Malignant'], autopct='%1.2f%%',colors=['green','red'])
    plt.show()

    # Splitting data: 67/33 train test
    trainX, testX, trainY, testY = train_test_split(data.data, data.target, test_size=0.33,random_state=42)

    # Build Naive Bayes model for binary classification
    model = GaussianNB()
    # Train model
    model.fit(trainX,trainY)

    # Use trained model to predict cancer cell classification
    pred = model.predict(testX)
    res = []
    for i in range(9):
        if pred[i]==1:
            res.append('Benign')
        else:
          res.append('Malignant')
    print(res)

    # Calculate accuracy 
    print(f"Model Accuracy: {accuracy_score(testY,pred)*100:.2f}%")