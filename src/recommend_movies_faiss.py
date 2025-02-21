import faiss
import numpy as np
import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer

# Set device
device = 'cpu'

# Load a smaller, faster embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

# Load dataset
df_filtered = pd.read_csv("assets/df_filtered.csv")

# Load embeddings
description_embeddings_np = np.load("assets/embeddings.npy")

# Load FAISS index
dimension = description_embeddings_np.shape[1]
index = faiss.read_index("assets/faiss_index.idx")

def recommend_movies(query: str, top_k: int = 5) -> pd.DataFrame:
    # Generate embedding for the query
    query_embedding = embedding_model.encode([query], convert_to_tensor=True, device=device).cpu().numpy()

    # Search in FAISS index
    distances, indices = index.search(query_embedding, top_k)

    # Retrieve recommended movies
    recommended_movies = df_filtered.iloc[indices[0]][['id', 'title', 'genres', 'vote_average', 'popularity', 'runtime']]
    
    return recommended_movies

if __name__ == "__main__":
    query = "Marvel movie with sci-fi action"
    recommendations = recommend_movies(query=query, top_k=15)
    print(recommendations[['title', 'genres']])
