import pandas as pd
import ast

def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df['description_length'] = df["overview"].str.split(" ").str.len()
    df['budget_revenue_ratio'] = df.apply(lambda row: row['revenue'] / row['budget'] if row['budget'] != 0 else None, axis=1)
    df['vote_share'] = df.apply(lambda row: row['vote_average'] * row['vote_count'], axis=1)
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df['year'] = df['release_date'].dt.year
    df['years_old'] = abs(df['year'] - 2023)
    return df

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