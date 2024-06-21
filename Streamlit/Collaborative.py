# Streamlit dependencies
import streamlit as st
from surprise import Dataset, Reader
from surprise import KNNBasic
from surprise.model_selection import train_test_split

# Data handling dependencies
import pandas as pd

# Importing data
@st.cache_data
def load_data():
    movies_df = pd.read_csv("TMBD Movie Dataset.csv")
    movies_df.drop(columns=['budget','revenue','cast','homepage','runtime','Unnamed: 0'], inplace=True)
    return movies_df

movies = load_data()

def collaborative_recommendation(original_title, k=10, test_size=0.1):
    # Load data into Surprise format
    reader = Reader(rating_scale=(0, 10))  # Assuming ratings are on a scale of 0 to 10
    data = Dataset.load_from_df(movies[['id', 'original_title', 'vote_average']], reader)

    # Split the data into training and testing sets
    trainset, testset = train_test_split(data, test_size=test_size)

    # Use the KNNBasic algorithm with cosine similarity and item-based approach
    algo = KNNBasic(sim_options={'name': 'cosine', 'user_based': False})  # Set user_based to False for item-based approach

    # Train the algorithm on the training set
    algo.fit(trainset)

    # Get the inner id of the selected movie title
    selected_movie_inner_id = trainset.to_inner_iid(original_title)

    # Get the top-N recommendations for the selected movie
    raw_recommended_movies = algo.get_neighbors(selected_movie_inner_id, k=k)

    # Convert inner ids back to movie titles
    recommended_movies = [trainset.to_raw_iid(inner_id) for inner_id in raw_recommended_movies]

    # Collect recommended movies info
    recommendations = []
    for movie_title in recommended_movies:
        movie_info = movies.loc[movies['original_title'] == movie_title, ['release_year', 'director', 'genres', 'tagline']].values[0]
        year = movie_info[0]
        director = movie_info[1]
        genres = movie_info[2]
        tagline = movie_info[3]
        recommendations.append((movie_title, year, director, genres, tagline))

    return recommendations

# Streamlit app
st.title('Movie Recommender System')

# Sidebar for movie selection and number of recommendations
selected_movie = st.sidebar.selectbox('Select a movie:', movies['original_title'])
num_recommendations = st.sidebar.slider('Number of Recommendations', min_value=1, max_value=10, value=5, step=1)

# Get recommendations based on selected movie and number of recommendations
recommended_movies = collaborative_recommendation(selected_movie, k=num_recommendations)

# Display recommended movies in a table
st.subheader(f"Recommended movies for '{selected_movie}':")

# Create a dataframe for recommended movies
recommended_df = pd.DataFrame(recommended_movies, columns=['Title', 'Year', 'Director', 'Genres', 'Tagline'])

# Adjust index to start from 1
recommended_df.index += 1

# Display the dataframe as a table
st.table(recommended_df)
