import pandas as pd
# import csv files as dataframes
beers = pd.DataFrame.read_csv('beers.csv')
brews = pd.DataFrame.read_csv('breweries.csv')

#create a merged version
merged = pd.merge(beers,brews, how='inner', left_on='brewery_id', right_on='id', sort=True, suffixes=('_beer', '_brew'))


if __name__ == '__main__':
    print(beers.dtypes)