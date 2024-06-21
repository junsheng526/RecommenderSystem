import streamlit as st
import pandas as pd
from Collaborative import collaborative_recommendation
from ContentBased import content_based_recommender
from Hybrid import hybrid_recommendation

# Load data and prepare it
@st.cache_data
def load_data():
    movies_df = pd.read_csv("TMBD Movie Dataset.csv")
    return movies_df

movies = load_data()

# Streamlit app
st.title("Movie Recommendation System")

# Sidebar for selecting recommendation system
st.sidebar.title("Select Recommendation System")
system_choice = st.sidebar.selectbox("", ("Content-Based Filtering", "Collaborative Filtering", "Hybrid Filtering"),index=0)

# Display selected model's UI and recommendation function
if system_choice == "Content-Based Filtering":
    st.subheader("Content-Based Filtering")
    title_key = "content_based_title"
    no_of_recommendations_key = "content_based_no_of_rec"
    title = st.selectbox("Select a Movie Title", movies['original_title'].unique(), key=title_key)
    no_of_recommendations = st.number_input("Number of Recommendations", min_value=1, value=5, key=no_of_recommendations_key)

    if title and no_of_recommendations:
        recommended_movies = content_based_recommender(title, no_of_recommendations)
    
elif system_choice == "Collaborative Filtering":
    st.subheader("Collaborative Filtering")
    selected_movie_key = "collaborative_selected_movie"
    num_recommendations_key = "collaborative_num_rec"
    selected_movie = st.sidebar.selectbox('Select a movie:', movies['original_title'], key=selected_movie_key)
    num_recommendations = st.sidebar.slider('Number of Recommendations', min_value=1, max_value=10, value=5, step=1, key=num_recommendations_key)

    # Call collaborative recommendation function and display recommendations
    if selected_movie and num_recommendations:
        recommended_movies = collaborative_recommendation(selected_movie, k=num_recommendations)

        # Display recommended movies in a table
        st.subheader(f"Recommended movies for '{selected_movie}':")

        # Create a dataframe for recommended movies
        recommended_df = pd.DataFrame(recommended_movies, columns=['Title', 'Year', 'Director', 'Genres', 'Tagline'])

        # Adjust index to start from 1
        recommended_df.index += 1

        # Display the dataframe as a table
        st.table(recommended_df)

elif system_choice == "Hybrid Filtering":
    st.subheader("Hybrid Filtering")
    selected_movie_key = "hybrid_selected_movie"
    selected_movie = st.sidebar.selectbox('Select a movie:', movies['original_title'], key=selected_movie_key)

    # Call hybrid recommendation function and display recommendations
    if selected_movie:
        hybrid_recommendation(selected_movie)
