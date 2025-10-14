import pandas as pd
import seaborn as sns
import matplotlib.pyplot

# import csv files as dataframes
beers = pd.read_csv('EDA/beers.csv')
brews = pd.read_csv('EDA/breweries.csv')

#create a merged version
merged = pd.merge(beers,brews, how='inner', left_on='brewery_id', right_on='id', sort=True, suffixes=('_beer', '_brew'))


#create a function to determine column categories in a dataframe
def  col_category(series):
    # determine unique number of categories
    unique = series.nunique(dropna=False)
    
    #determine total count
    total = len(series)
    
    # return correct categories
    if pd.api.types.is_numeric_dtype(series):
        return 'Numeric'
    elif pd.api.types.is_datetime64_any_dtype(series):
        return 'Date'
    elif unique == total:
        return 'Text (Unique)'
    else:
        return 'Categorical'
    
# function to print dataframe categories
def print_categories(df):
    for col in df.columns:
        print(col, ": ", col_category(df[col]))
        
        
        
        
def descriptive_stats():
    #determine number of numerical values in columns in series
    lIbu = len(beers['ibu'])
    lAbv = len(beers['abv'])
    lOunces = len(beers['ounces'])
    print('Lengths of ibu', lIbu)
    print('Lengths of abv', lAbv)
    print('Lengths of ounces', lOunces)
    print('-----------------------')
    
    #count number of non-missing values in each numerical column
    cIbu = beers['ibu'].count()
    cAbv = beers['abv'].count()
    cOunces = beers['ounces'].count()
    print('Counts of non-null ibu', cIbu)
    print('Counts of  non-null abv', cAbv)
    print('Counts of non-null ounces', cOunces)
    print('-----------------------')
    
    #calculate the number of missing values in each numerical column
    pct_mIbu = "{0:.1f}%".format(((float(lIbu - cIbu))/lIbu)*100)
    pct_mAbv = "{0:.1f}%".format(((float(lAbv - cAbv))/lAbv)*100)
    pct_mOunces = "{0:.1f}%".format(((float(lOunces - cOunces))/lOunces)*100)
    print('Percentage of missing ibu', pct_mIbu)
    print('Percentage of missing abv', pct_mAbv)
    print('Percentage of missing ounces', pct_mOunces)
    print('-----------------------')
    
    #determine min and max values in each numerical column
    print('Min ibu', beers['ibu'].min(), 'Max ibu', beers['ibu'].max())
    print('Min abv', beers['abv'].min(), 'Max abv', beers['abv'].max())
    print('Min ounces', beers['ounces'].min(), 'Max ounces', beers['ounces'].max())
    print('-----------------------')
    
    #calculate mode of each numerical column (mode = most frequent value)
    print('Mode ibu', beers['ibu'].mode())
    print('Mode abv', beers['abv'].mode())
    print('Mode ounces', beers['ounces'].mode())
    print('-----------------------')
    
    #calculate the mean of each numerical column (mean = average)
    print('Mean ibu', beers['ibu'].mean())
    print('Mean abv', beers['abv'].mean())
    print('Mean ounces', beers['ounces'].mean())
    print('-----------------------')
    
    #calculate the median of each numerical column (median = middle value)
    print('Median ibu', beers['ibu'].median())
    print('Median abv', beers['abv'].median())
    print('Median ounces', beers['ounces'].median())
    print('-----------------------')
    
    #calculate the standard deviation of each numerical column (standard deviation = spread)
    print('Standard deviation ibu', beers['ibu'].std())
    print('Standard deviation abv', beers['abv'].std())
    print('Standard deviation ounces', beers['ounces'].std())
    print('-----------------------')
    
    #quantile statistics (cut points that split the data into equal sized groups)
    qIbu = beers['ibu'].quantile([0.25, 0.5, 0.75])
    qAbv = beers['abv'].quantile([0.25, 0.5, 0.75])
    qOunces = beers['ounces'].quantile([0.25, 0.5, 0.75])
    print('Quantiles ibu:\n', qIbu)
    print('Quantiles abv:\n', qAbv)
    print('Quantiles ounces:\n', qOunces)
    print('-----------------------')


#create a frequency distribution plot of numerical columns in beers
def plot_beers():
   sns.displot(beers['ibu'].dropna(),kde=True)
   matplotlib.pyplot.show()
   
   sns.displot(beers['abv'].dropna(),kde=True)
   matplotlib.pyplot.show()
   
   sns.displot(beers['ounces'].dropna(),kde=True)
   matplotlib.pyplot.show()


def correlation():
   cors = beers[['abv','ibu','ounces']].corr()
   print(cors)

def desc():
    print(beers[['name','style']].describe())

if __name__ == '__main__':
    print_categories(beers)
    print("-----------------------")
    print_categories(brews)
    print("-----------------------")
    print('Determining descriptive statistics of numerical values of beers')
    descriptive_stats()
    print('-----------------------')
    print('Plotting frequency distributions of numerical values of beers')
    plot_beers()
    print('-----------------------')
    
    print('Determining correlations between numerical values using Pearson\'s correlation coefficient')
    correlation()
    print ('---------------------')
    
    print('Determining descriptive statistics of categorical values of beers')
    desc()
    print ('---------------------')

