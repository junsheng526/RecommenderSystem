# Script dependencies
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Importing data
movies_df = pd.read_csv('TMBD Movie Dataset.csv')
link_df = pd.read_csv('links_small.csv')
ratings_df = pd.read_csv('ratings_small.csv')
ratings_df.drop(['timestamp'], axis=1,inplace=True)

merged_df = pd.merge(ratings_df, link_df, on='movieId', how='inner')
merged_df = pd.merge(merged_df, movies_df,left_on='tmdbId', right_on='id', how='inner')
titles = movies_df[['original_title']]
merged_df.drop(columns=['movieId','imdbId','tmdbId','Unnamed: 0'],inplace=True)
merged_df.rename(columns={'id': 'movieId'}, inplace=True)

indices=pd.Series(data=list(titles.index), index= titles['original_title'] )


def collab_model(original_title):

    moviemat = merged_df.pivot_table(index='userId', columns='original_title', values='rating')
    
    movie_ratings = moviemat[original_title]
    # Calculate correlation with other movies
    similar_to_movie = moviemat.corrwith(movie_ratings)
    corr_movie = pd.DataFrame(similar_to_movie, columns=['Correlation'])
    corr_movie.dropna(inplace=True)
    corr_movie = corr_movie.sort_values(by='Correlation', ascending=False)

    corr_movie = corr_movie[1:10+1]

    # Get the movie indices
    movie_indices = corr_movie.index

    recommendations = []
    for movie_title in movie_indices:
        year = merged_df.loc[merged_df['original_title'] == movie_title, 'release_year'].values[0]
        recommendations.append((movie_title, year))  
        
    return recommendations
