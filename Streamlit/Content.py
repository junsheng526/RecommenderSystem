import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

@st.cache_data
def load_data():
    data = pd.read_csv("TMBD Movie Dataset.csv")
    return data

movies = load_data()

def format_names(x):
    # Check if the input is a string and contains '|'
    if isinstance(x, str) and '|' in x:
        # Split the string by '|', strip whitespace, and return as a list
        return [name.strip() for name in x.split('|')]
    else:
        # If input is not a string or doesn't contain '|', return an empty list
        return []
    
# Specify columns to process
features = ['cast', 'keywords', 'genres']

# Apply the format_names function to each specified feature column
for feature in features:
    movies[feature] = movies[feature].apply(format_names)

def format_director(x):
    # Check if the input is a non-empty string
    if isinstance(x, str) and x.strip():  # Ensure the string is not empty after stripping whitespace
        return [x.strip()]  # Return a list with the stripped director name
    else:
        return []  # Return an empty list for invalid or empty input

# Apply the format_director function to the 'director' column
movies['director'] = movies['director'].apply(format_director)

def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        #Check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''
        
features = ['cast', 'keywords', 'director', 'genres']

for feature in features:
    movies[feature] = movies[feature].apply(clean_data)

def create_soup(row):
    # Join keywords, cast, and genres into a single string
    soup_parts = []
    
    # Append keywords (if available)
    if isinstance(row['keywords'], list):
        soup_parts.extend(row['keywords'])
    
    # Append cast (if available)
    if isinstance(row['cast'], list):
        soup_parts.extend(row['cast'])
    
    # Append director (if available)
    if isinstance(row['director'], str):
        soup_parts.append(row['director'])
    
    # Append genres (if available)
    if isinstance(row['genres'], list):
        soup_parts.extend(row['genres'])
    
    # Join all soup parts into a single string
    return ' '.join(soup_parts)

# Apply the create_soup function to each row of the DataFrame along axis=1
movies['soup'] = movies.apply(create_soup, axis=1)

@st.cache_data
def calculate_cosine_sim():
    # Calculate the linear kernel using the TF-IDF matrix
    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(movies['soup'])
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    return cosine_sim

# Calculate and cache the linear kernel
cosine_sim = calculate_cosine_sim()

titles = movies[['original_title']]
indices=pd.Series(data=list(titles.index), index= titles['original_title'] )

def content_model(title):
    # Get the index of the movie that matches the title
    index = indices[title]
    
    # Get the pairwsie similarity scores of all movies with the selected movie
    sim_scores = list(enumerate(cosine_sim[index]))
    
    # Sort the movies based on the similarity scores in descending order
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the scores of the top n most similar movies 
    sim_scores = sim_scores[1:10+1]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    recommendations = []
    for (year, title) in (movies.iloc[movie_indices][['release_year', 'original_title']].values):
        recommendations.append((title, year))
    
    return recommendations


