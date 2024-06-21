
# Streamlit dependencies
import streamlit as st

# Data handling dependencies
import pandas as pd
import numpy as np

# Custom Libraries
from Collab import collab_model
from Content import content_model
import numpy as np

@st.cache_data
def load_data():
    movies_df = pd.read_csv("TMBD Movie Dataset.csv")
    return movies_df

movies = load_data()

movie_titles = movies[['original_title']]




# Display hybrid recommendations to the user
def hybrid_recommendation(original_title):
    st.subheader(f"Recommendations for Title: {original_title}")

    # Call the hybrid recommender function to get recommendations
    content_recommendations = content_model(original_title)
    collab_recommendations = collab_model(original_title)
    recommendations = content_recommendations + collab_recommendations

    if recommendations is None or len(recommendations) == 0:
        st.warning("No recommendations available for this title.")
    else:
        # Display recommendations
        for i, (original_title, year) in enumerate(content_recommendations, start=1):
            st.write(f"{i}. {original_title} ({year}) ")
        
        st.subheader(f"Users Also Like : ")
        for i, (original_title, year) in enumerate(collab_recommendations, start=1):
            st.write(f"{i}. {original_title} ({year})")

# Title and Sidebar
st.title("Hybrid Movies Recommender System")
st.sidebar.header("User Input")

# User Input for Movie Title and Number of Recommendations
original_title = st.sidebar.selectbox("Select a Movie Title", movies['original_title'].unique())


if original_title == '':
    st.warning(f"Title is null")
else:
    # Display hybrid recommendations for the selected movie title
    hybrid_recommendation(original_title)
