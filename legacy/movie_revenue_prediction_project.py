import streamlit as st
import pandas as pd
import numpy as np
import joblib  # For saving/loading ML model
from sklearn.linear_model import LinearRegression

# Load the dataset
df = pd.read_csv("tmdb_movies_data.csv")

# Prepare data
X = df[['budget', 'popularity']]
y = df['revenue']

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
joblib.dump(model, "movie_revenue_model.pkl")

# Load model
model = joblib.load("movie_revenue_model.pkl")

# Streamlit UI
st.title("Movie Revenue Prediction")
st.write("Enter the budget and popularity to predict revenue:")

# User input fields
budget = st.number_input("Movie Budget", min_value=0, value=1000000)
popularity = st.number_input("Popularity Score", min_value=0.0, value=10.0)

# Predict button
if st.button("Predict Revenue"):
    prediction = model.predict(np.array([[budget, popularity]]))
    st.success(f"Predicted Revenue: ${prediction[0]:,.2f}")


import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("tmdb_movies_data.csv")

# TMDB API Key (Replace with your own key)
API_KEY = "your_tmdb_api_key"

def get_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}"
    data = requests.get(url).json()
    return "https://image.tmdb.org/t/p/w500" + data.get('poster_path', '')

# Recommendation System
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(df['overview'].fillna(''))
similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)

def recommend(movie_title):
    if movie_title not in df['original_title'].values:
        return []
    idx = df[df["original_title"] == movie_title].index[0]
    similar_movies = sorted(list(enumerate(similarity[idx])), key=lambda x: x[1], reverse=True)[1:6]
    return [(df.iloc[i[0]]['title'], df.iloc[i[0]]['id']) for i in similar_movies]

# Streamlit UI
st.title("Movie Recommendation System")
movie_choice = st.selectbox("Pick a movie", df["original_title"].unique())

if st.button("Recommend"):
    recommendations = recommend(movie_choice)
    if recommendations:
        for title, movie_id in recommendations:
            st.write(title)
            st.image(get_poster(movie_id))
    else:
        st.write("No recommendations found.")


import os
import requests
from dotenv import load_dotenv

# Load API Key from .env file
load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

# Check if API Key is loaded correctly
if not TMDB_API_KEY:
    print("API Key not found! Make sure it's set in the .env file.")
    exit()

# Example function to fetch movie details using TMDB API
def get_movie_details(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}"
    response = requests.get(url)
    
    if response.status_code == 200:
        return response.json()  # Return movie details as JSON
    else:
        return None  # Return None if API request fails

# Test API Call
movie_id = 550  # Example: Fight Club
movie_data = get_movie_details(movie_id)

if movie_data:
    print("Movie Title:", movie_data["title"])
    print("Overview:", movie_data["overview"])
else:
    print("Failed to fetch movie details.")
