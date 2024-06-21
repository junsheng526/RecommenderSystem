import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from imdb import IMDb  # Correct import for IMDbPY

# Function to load movie data
@st.cache_data
def load_data():
    data = pd.read_csv("TMBD Movie Dataset.csv")
    return data

# Load movie data
movies = load_data()

# Function to format names
def format_names(x):
    if isinstance(x, str) and '|' in x:
        return [name.strip() for name in x.split('|')]
    else:
        return []

# Specify columns to process
features = ['cast', 'keywords', 'genres']

# Apply the format_names function to each specified feature column
for feature in features:
    movies[feature] = movies[feature].apply(format_names)

# Function to clean data
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''

# Apply the clean_data function to specified columns
features = ['cast', 'keywords', 'director', 'genres']
for feature in features:
    movies[feature] = movies[feature].apply(clean_data)

# Function to create soup
def create_soup(row):
    soup_parts = []
    if isinstance(row['keywords'], list):
        soup_parts.extend(row['keywords'])
    if isinstance(row['cast'], list):
        soup_parts.extend(row['cast'])
    if isinstance(row['director'], str):
        soup_parts.append(row['director'])
    if isinstance(row['genres'], list):
        soup_parts.extend(row['genres'])
    return ' '.join(soup_parts)

# Apply the create_soup function to each row of the DataFrame along axis=1
movies['soup'] = movies.apply(create_soup, axis=1)

# Function to calculate cosine similarity
@st.cache_data
def calculate_cosine_sim():
    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(movies['soup'])
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    return cosine_sim

# Calculate cosine similarity matrix
cosine_sim = calculate_cosine_sim()

# Get movie titles and create indices
titles = movies[['original_title']]
indices = pd.Series(data=list(titles.index), index=titles['original_title'])

# Function to fetch movie details from IMDb
def get_movie_details(title):
    ia = IMDb()
    search_results = ia.search_movie(title)
    
    if search_results:
        movie_id = search_results[0].movieID
        movie = ia.get_movie(movie_id)
        
        return {
            'title': movie.get('title', ''),
            'year': movie.get('year', ''),
            'director': [str(person) for person in movie.get('directors', [])],
            'poster_url': movie.get('full-size cover url', '')
        }
    
    return None

# Function to perform content-based recommendation
def content_based_recommender(title, no_of_recommendations):
    if title not in indices:
        return []

    index = indices[title]
    sim_scores = list(enumerate(cosine_sim[index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:no_of_recommendations+1]

    # Extract movie indices from similarity scores
    movie_indices = [i[0] for i in sim_scores]

    recommendations = []
    for (director, year, title) in movies.iloc[movie_indices][['director', 'release_year', 'original_title']].values:
        recommendations.append((title, year, director))
    
    return recommendations

# Streamlit app
def main():
    st.title("Movies Recommender System")
    st.sidebar.header("User Input")

    title = st.sidebar.selectbox("Select a Movie Title", movies['original_title'].unique())
    no_of_recommendations = st.sidebar.number_input("Number of Recommendations", min_value=1, value=5)

    if title == '':
        st.warning("Please select a movie title.")
    else:
        recommendations = content_based_recommender(title, no_of_recommendations)
        if recommendations:
            st.subheader(f"Recommendations for '{title}':")
            for i, (movie_title, year, director) in enumerate(recommendations, start=1):
                movie_details = get_movie_details(movie_title)
                if movie_details:
                    st.write(f"{i}. {movie_title} ({year}) by {', '.join(director)}")
                    st.image(movie_details['poster_url'], caption=movie_title, width=200)
                else:
                    st.write(f"{i}. {movie_title} ({year}) by {', '.join(director)} (Poster not available)")
        else:
            st.warning("No recommendations available.")

if __name__ == "__main__":
    main()
