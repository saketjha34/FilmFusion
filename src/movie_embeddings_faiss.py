import faiss
import numpy as np
import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer

def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df['description_length'] = df["overview"].str.split(" ").str.len()
    df['budget_revenue_ratio'] = df.apply(lambda row: row['revenue'] / row['budget'] if row['budget'] != 0 else None, axis=1)
    df['vote_share'] = df.apply(lambda row: row['vote_average'] * row['vote_count'], axis=1)
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df['year'] = df['release_date'].dt.year
    df['years_old'] = abs(df['year'] - 2023)
    return df
# Load model

device = 'cpu'
print(device)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

# Load dataset
df = pd.read_csv("dataset/TMDB10K.csv")
df_filtered = preprocess_dataset(df)

# Fix NaN issues before encoding
df_filtered['overview'] = df_filtered['overview'].fillna("")  
df_filtered['genres'] = df_filtered['genres'].fillna("[]")    
df_filtered ['overview_genre_combined'] = df_filtered['genres'].apply(lambda x: ' '.join(x))+ ' ' + df_filtered['overview']

# Compute embeddings
description_embeddings = embedding_model.encode(df_filtered['overview_genre_combined'].tolist(), convert_to_tensor=True, device=device)
description_embeddings_np = description_embeddings.cpu().detach().numpy()

# Save embeddings to a file
np.save("assets/embeddings.npy", description_embeddings_np)

# Create and save FAISS index
dimension = description_embeddings_np.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(description_embeddings_np)
faiss.write_index(index, "assets/faiss_index.idx")

# Save dataframe (filtered) with indexes
df_filtered.to_csv("df_filtered.csv", index=False)

model_dict = {
    "model": embedding_model, 
    "tokenizer": embedding_model.tokenizer
}

# Save to a file
if __name__ == "__main__":
    joblib.dump(model_dict, "assets/embedding_model.joblib")
    print("Embeddings, FAISS index, and dataset saved successfully!")