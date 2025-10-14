import pandas as pd
import seaborn as sns
import matplotlib.pyplot

iris = pd.read_csv('iris.csv')

if __name__ == '__main__':
    print(iris.head())
    print('-----------------------')