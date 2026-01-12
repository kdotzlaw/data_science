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
INPUT: row in dataframe
OUTPUT: lowercase features with no spaces
PROCESS:

'''

def clean(row):
    if isinstance(row,list):
        return [str.lower(i.replace(" ","")) for i in row]
    else:
        if isinstance(row, str):
            return str.lower(row.replace(" ",""))
        else:
            return ""
        
'''
INPUT: list of features
OUTPUT: a soup of metadata
PROCESS: join metadata together with ' ' separator

'''
def soup(features):
     return ' '.join(features['keywords']) + ' ' + ' '.join(features['cast']) + ' ' + features['director'] + ' ' + ' '.join(features['genres'])


'''
INPUT: movies dataframe,movie indices, movie title (str), similarity scores (list)
OUTPUT: list of movie titles
PROCESS:
    1. Get index of movie by title
    2. get list of similarity scores of all movies
    3. create list of tuples (index, similarity score)
    4. sort list desc based on similarity score
    5. get top 10 from list (exclude 1st element -- title)
    6. map indices to titles and return movie list
'''
def recommendations(movies,indices, title, cosSim):
    ids = indices[title]
    scores = list(enumerate(cosSim[ids]))
    # sort desc
    scores = sorted(scores, key=lambda x:x[1],reverse=True)

    # create list of tuples (index, score)
    scores = scores[1:11]

    mIndex = [index[0] for index in scores]
    result = movies["original_title"].iloc[mIndex]
    return result



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
    #print(movies[['original_title', 'cast', 'director', 'keywords', 'genres']].head())

    # clean row data -- only lowercase, no spaces
    features = ['cast','keywords','director','genres']
    for feature in features:
        movies[feature] = movies[feature].apply(clean)

    # create a soup of all metadata
    movies['soup'] = movies.apply(soup,axis=1)
    print(movies['soup'].head())

    # preprocess data & convert into vector with CountVectorizer
    # ignore stopWords (an, a, the...)
    countVector = CountVectorizer(stop_words='english')
    countMatrix = countVector.fit_transform(movies['soup'])
    print(countMatrix.shape)

    # use Cosine similarity to score matrix
    cos = cosine_similarity(countMatrix,countMatrix)
    print(cos.shape)
    

    # reset index of dataframe 
    movies = movies.reset_index()

    # create reverse mapping of titles to indices
    indices = pd.Series(movies.index,index=movies["original_title"]).drop_duplicates()
    print(indices.head())

    print("---------------Content Based Movie Recommendations----------------")
    print("Recommendations for Indiana Jones and the Temple of Doom")
    print(recommendations(movies,indices,title="Indiana Jones and the Temple of Doom",cosSim=cos))
    print()
    print("Recommendations for The Avengers")
    print(recommendations(movies,indices,title="The Avengers",cosSim=cos))

  

if __name__ == '__main__':
    # create dataframes for exploratory data analysis
    credits = pd.read_csv("tmdb_5000_credits.csv")
    movies = pd.read_csv("tmdb_5000_movies.csv")

    # Only need id, title, cast, crew of credits -- merge on id with movies
    movies = movies.merge(credits,on='id')
    mrsystem(movies)