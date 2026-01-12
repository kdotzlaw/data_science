'''
A Content-Based recommendation system for movies
'''
import pandas as pd


'''
INPUT: 2 dataframes - credits, movies
OUTPUT:
PROCESS:
'''
def eda(credits, movies):
    print("Viewing df heads...")
    print(movies.head())
    print(credits.head())

    # Need


if __name__ == '__main__':
    # create dataframes for exploratory data analysis
    credits = pd.read_csv("tmdb_5000_credits.csv")
    movies = pd.read_csv("tmdb_5000_movies.csv")

    # explore data
    eda(credits, movies)