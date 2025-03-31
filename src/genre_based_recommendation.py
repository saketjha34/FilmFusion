import pandas as pd
from utils import preprocess_dataset, recommend_movies_by_genre


if __name__ == "__main__":
    df = pd.read_csv("dataset/TMDB10K.csv")
    df = preprocess_dataset(df)
    genre_query = "Action, Thriller"
    sort_by = "vote_share"
    years_old = 6
    language = "English"
    k = 10
    recommendations = recommend_movies_by_genre(df, genre_query, sort_by, years_old, language, k)
    print(recommendations)