import faiss
import numpy as np
import pandas as pd
import joblib

# Set device
device = 'cpu'

# Load the saved model using joblib
model_dict = joblib.load("assets/embedding_model.joblib")

# Extract the model and tokenizer
embedding_model = model_dict["model"]
tokenizer = model_dict["tokenizer"]
embedding_model.tokenizer = tokenizer  # Ensure tokenizer is correctly assigned

# Load dataset
df_filtered = pd.read_csv("assets/df_filtered.csv")

# Load embeddings
description_embeddings_np = np.load("assets/embeddings.npy")

# Load FAISS index
dimension = description_embeddings_np.shape[1]
index = faiss.read_index("assets/faiss_index.idx")

def recommend_movies(query: str, top_k: int = 5) -> pd.DataFrame:
    # Generate embedding for the query
    query_embedding = embedding_model.encode([query], convert_to_tensor=True, device=device).cpu().detach().numpy()

    # Search in FAISS index
    distances, indices = index.search(query_embedding, top_k)

    # Retrieve recommended movies
    recommended_movies = df_filtered.iloc[indices[0]][['id', 'title', 'genres', 'vote_average', 'popularity', 'runtime']]
    
    return recommended_movies

if __name__ == "__main__":
    query = "Marvel movie with scifi action"
    recommend_movies = recommend_movies(query= query, top_k=15)
    print(recommend_movies[['title', 'genres']])
