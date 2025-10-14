import pandas as pd
import seaborn as sns
import matplotlib.pyplot

iris = pd.read_csv('Iris_Classification/Iris.csv')



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
    #create multi plots of numerical columns and color code by species
   g = sns.PairGrid(iris)
   g.map(sns.scatterplot, hue=iris['Species'])
   g.add_legend()
   matplotlib.pyplot.show() 
    
    
def plots_multi():
    g = sns.PairGrid(iris)
    g.map_upper(sns.scatterplot, hue=iris['Species'])
    g.map_lower(sns.kdeplot)
    g.map_diag(sns.kdeplot, lw=3, legend=False)
    matplotlib.pyplot.show()

def correlation():
    #create correlation heatmap
    cors = iris[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']].corr()
    matplotlib.pyplot.figure(figsize=(10,4))
    sns.heatmap(cors, annot=True, cmap='Reds_r')
    matplotlib.pyplot.title('Correlation Matrix')
    matplotlib.pyplot.show()
    
if __name__ == '__main__':
    descriptive_stats()
    print('-----------------------')
    
    plots_scatter()
    
    plots_multi()
    
    correlation()
    print ('---------------------')