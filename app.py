import streamlit as st
import pandas as pd
import numpy as np
import ast
import faiss
from sentence_transformers import SentenceTransformer
import time  
device = 'cpu'

# Page Configurations
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")


# Cached function (without progress bar)
@st.cache_resource
def load_data():
    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    df = pd.read_csv("dataset/TMDB10K.csv")
    embeddings = np.load("assets/embeddings.npy")
    index = faiss.read_index("assets/faiss_index.idx")
    return df, index, embeddings, model

# Progress Bar & UI Enhancements
with st.spinner("🔄 Enhancing your experience... Loading model and dataset. Please wait."):
    progress_bar = st.progress(0)

    # Simulate loading progress
    time.sleep(0.5)
    progress_bar.progress(10)
    
    st.session_state.df, st.session_state.index, st.session_state.embeddings, st.session_state.model = load_data()

    progress_bar.progress(100)

st.success("✅ Data Loaded Successfully!")
time.sleep(1)  # Let message be visible


# Preprocess Dataset
def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df['description_length'] = df["overview"].str.split(" ").str.len()
    df['budget_revenue_ratio'] = df.apply(lambda row: row['revenue'] / row['budget'] if row['budget'] != 0 else None, axis=1)
    df['vote_share'] = df.apply(lambda row: row['vote_average'] * row['vote_count'], axis=1)
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df['year'] = df['release_date'].dt.year
    df['years_old'] = abs(df['year'] - 2023)
    return df

st.session_state.df = preprocess_dataset(st.session_state.df)

# Recommendation Function by Genre
def recommend_movies_by_genre(df: pd.DataFrame, genre_query: str, sort_by: str, years_old: int, language: str, k: int) -> pd.DataFrame:
    def safe_convert(val):
        if isinstance(val, str):
            try:
                return ast.literal_eval(val) if "[" in val else [val]
            except:
                return []
        elif isinstance(val, list):
            return val
        else:
            return []
    
    df['genres'] = df['genres'].apply(safe_convert)
    genre_list = [g.strip() for g in genre_query.split(",")]
    filtered_df = df[df['genres'].apply(lambda genres: isinstance(genres, list) and any(g in genres for g in genre_list))]
    filtered_df = filtered_df[filtered_df['years_old'] <= years_old]
    filtered_df = filtered_df[filtered_df['original_language'] == language]
    sorted_df = filtered_df.sort_values(by=[sort_by], ascending=False)

    return sorted_df[['id', 'title', 'genres', 'vote_average', 'popularity', 'runtime', 'budget_revenue_ratio']].head(k)


def recommend_movies_by_query(query: str, top_k: int = 5) -> pd.DataFrame:
    # Generate embedding for the query
    query_embedding = st.session_state.model.encode([query], convert_to_tensor=True, device=device).cpu().detach().numpy()

    # Search in FAISS index
    distances, indices = st.session_state.index.search(query_embedding, top_k)

    # Retrieve recommended movies
    recommended_movies = st.session_state.df.iloc[indices[0]][['id', 'title', 'genres', 'vote_average', 'popularity', 'runtime']]
    
    return recommended_movies


# Title
st.title("Movie Recommender System 🎬")
st.subheader("Find the best movies based on your preferences!", divider=True)


# Initialize session state variables
if "genre_recommendations" not in st.session_state:
    st.session_state.genre_recommendations = None
if "query_recommendations" not in st.session_state:
    st.session_state.query_recommendations = None
if "last_clicked" not in st.session_state:
    st.session_state.last_clicked = None

# Genre-Based Recommendation Container
with st.container(border=True):
    st.subheader("🎭 Genre-Based Movie Recommendations")
    
    genre_query = st.text_input("Enter Genre(s) (comma-separated)", placeholder="Science Fiction, Crime", key="genre_query")
    sort_by = st.selectbox("Sort By", ["runtime", "vote_share", "popularity", "budget_revenue_ratio"], key="sort_by")
    years_old = st.slider("Maximum Years Old", 0, 120, 6, key="years_old")
    language = st.selectbox("Select Language", ['English', 'Spanish', 'Finnish', 'Polish', 'German', 'Korean',
       'Chinese', 'Japanese', 'French', 'Dutch', 'Portuguese', 'Italian',
       'Danish', 'Tagalog', 'Ukrainian', 'Russian', 'Norwegian',
       'Romanian', 'Tamil', 'Swedish', 'Telugu', 'Icelandic',
       'Turkish', 'Basque', 'Indonesian', 'Thai', 'Macedonian',
       'Arabic', 'Serbian', 'Hindi', 'Vietnamese', 'Bulgarian',
       'Galician', 'Greek', 'Persian', 'Catalan', 'Czech',
       'Malayalam', 'Irish', 'Hebrew', 'Hungarian'], key="language")
    k = st.slider("Number of Recommendations", 1, 1000, 15, key="k")

    if st.button("Get Recommendations by Genre", key="genre_button"):
        st.session_state.genre_recommendations = recommend_movies_by_genre(
            st.session_state.df, genre_query, sort_by, years_old, language, k
        )
        st.session_state.last_clicked = "genre"

# Query-Based Recommendation Container
with st.container(border=True):
    st.subheader("🔍 Query-Based Movie Recommendations")

    query = st.text_input("Enter a movie description to find similar movies", 
                          placeholder="A heist thriller with a genius mastermind", key="query_text")
    top_k_query = st.number_input("Number of Query-Based Recommendations", 
                                  min_value=1, max_value=1000, value=10, step=1, key="top_k_query")

    if st.button("Get Recommendations by Query", key="query_button"):
        st.session_state.query_recommendations = recommend_movies_by_query(query, top_k_query)
        st.session_state.last_clicked = "query"

# Results Display Container (Only Show Latest Button Clicked)
with st.container(border=True):
    if st.session_state.last_clicked == "genre" and st.session_state.genre_recommendations is not None:
        st.subheader("🎬 Genre-Based Recommendations")
        print(st.session_state.genre_recommendations)
        st.write(st.session_state.genre_recommendations)

    elif st.session_state.last_clicked == "query" and st.session_state.query_recommendations is not None:
        st.subheader("🎬 Query-Based Recommendations")
        st.write(st.session_state.query_recommendations)