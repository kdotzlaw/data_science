'''
A Content-Based recommendation system for movies
- personalize predictions using plot details
- use movie metadata to improve predictions
'''
import pandas as pd
from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


'''
INPUT: movie entry in a dataframe
OUTPUT: Director name (str) or NAN if left empty
PROCESS: searches thru movie data to find job==director
'''
def getDirector(movie):
    for i in movie:
        if i['job']=='Director':
            return i['name']
    return np.nan


'''
INPUT: movie entry in a dataframe
OUTPUT: top 3 elements in list 
PROCESS: 
'''
def getList(movie):
    if isinstance(movie,list):
        names=[i['name'] for i in movie]
        if len(names) > 3:
            names=names[:3]
        return names
    return []


'''
INPUT: movies dataframe
OUTPUT:
PROCESS:
- Uses cast, crew (director), keywords, genres to personalize recommendation
'''
def mrsystem(movies):
    features = ['cast','crew','keywords','genres']
    # convert features into literals
    for feature in features:
        movies[feature]=movies[feature].apply(literal_eval)

    # create new columns for features
    movies['director']=movies['crew'].apply(getDirector)
    features = ['cast','keywords','genres']
    for feature in features:
        movies[feature]=movies[feature].apply(getList)
    print(movies[['title', 'cast', 'director', 'keywords', 'genres']].head())

if __name__ == '__main__':
    # create dataframes for exploratory data analysis
    credits = pd.read_csv("tmdb_5000_credits.csv")
    movies = pd.read_csv("tmdb_5000_movies.csv")

    # explore data
    print("Viewing df heads...")
    print(movies.head())
    print(credits.head())

    # Only need id, title, cast, crew of credits -- merge on id with movies
    movies = movies.merge(credits,on='id')
    mrsystem(movies)