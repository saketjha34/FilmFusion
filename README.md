# FilmFusion - AI-Powered Movie Recommender System 🎬

## 🌟 Live Demo
🚀 **Try it out here**: [FilmFusionAI](https://filmfusionai.streamlit.app/)

---

## 📌 Overview
FilmFusion is an advanced **AI-powered movie recommendation system** that helps users find movies based on **genres, descriptions, and preferences**. It leverages **FAISS for fast similarity search**, **Sentence Transformers for text-based recommendations**, and an extensive movie dataset to provide accurate suggestions.

---

## 🚀 Features

### 🎭 **Genre-Based Recommendations**
- Select multiple genres from a predefined list.
- Filter results based on movie popularity, runtime, or budget-to-revenue ratio.
- Limit recommendations based on the movie's age.
- Supports multiple languages.

### 🔍 **Query-Based Recommendations**
- Input a natural language query (e.g., "A sci-fi movie with space battles").
- Uses Sentence Transformers (`all-MiniLM-L6-v2`) to generate movie embeddings.
- Searches efficiently using FAISS (Facebook AI Similarity Search).

### 📊 **Data Enhancements & Processing**
- **Metadata Enrichment:** Calculates additional features like:
  - Description length
  - Budget-to-revenue ratio
  - Vote share (vote average × vote count)
  - Years since release
- **Efficient Search:** Uses FAISS indexing for fast similarity search.
- **Interactive UI:** Built with Streamlit for a seamless experience.

---

## 🛠️ Tech Stack

- **Frontend & Deployment:** [Streamlit](https://streamlit.io/)
- **Data Processing:** Pandas, NumPy, Pytorch
- **NLP Model:** [SentenceTransformers](https://www.sbert.net/)
- **Efficient Search:** FAISS (Facebook AI Similarity Search)
- **Storage & Indexing:**
  - Movie dataset (`TMDB10K.csv`)
  - Precomputed embeddings (`embeddings.npy`)
  - FAISS index (`faiss_index.idx`)

---

## 📦 Installation & Setup

1. **Clone the repository:**
   ```sh
   git clone https://github.com/saketjha34/FilmFusion.git
   cd FilmFusion
   ```

2. **Create a virtual environment & activate it:**
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

4. **Run the app locally:**
   ```sh
   streamlit run app.py
   ```

---

## 📂 Project Structure
```
FilmFusion/
│── dataset/
│   └── TMDB10K.csv  # Movie dataset
│── assets/
│   ├── embeddings.npy  # Precomputed embeddings
│   ├── faiss_index.idx  # FAISS index
│   ├── df_filtered.csv  # Filtered df
│   ├── embedding_model.joblib # saved embedding model
│── src/
│   ├── movie_embeddings_faiss.py  # Preprocess Data and Generate Embeddings
│   ├── recommend_movies_faiss.py # Basic CLI Movie Recommender
│── images/
│   ├── images  # images and assets
│── app.py  # Main Streamlit app
│── requirements.txt  # Dependencies
│── README.md  # Project documentation
│── .gitignore  # git ignore
```

---

## 📈 How It Works

### 🔹 **Data Processing**
- The dataset (`TMDB10K.csv`) is loaded and preprocessed.
- Genres are converted into lists for easy filtering.
- Additional metadata features are computed.

### 🔹 **Genre-Based Recommendation**
1. User selects genres, sorting preference, language, and max movie age.
2. The system filters the dataset based on the selected criteria.
3. The top `k` recommendations are displayed.

### 🔹 **Query-Based Recommendation**
1. The user inputs a description (e.g., "A thrilling heist movie").
2. The input is converted into an embedding using `all-MiniLM-L6-v2`.
3. FAISS searches for the most similar movies.
4. The top `k` closest matches are displayed.

---

## 📖 References
- **FAISS**: [https://faiss.ai/](https://faiss.ai/)
- **Sentence Transformers**: [https://www.sbert.net/](https://www.sbert.net/)
- **TMDB Movie Dataset**: [https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
- **Streamlit**: [https://streamlit.io/](https://streamlit.io/)
- **Python Pandas**: [https://pandas.pydata.org/](https://pandas.pydata.org/)
- **NumPy**: [https://numpy.org/](https://numpy.org/)

---

## 🛠️ Future Improvements
✅ Improve embedding quality using larger models (e.g., `sentence-transformers/msmarco-distilbert-base-v4`)
✅ Add user personalization & collaborative filtering
✅ Enhance UI with movie posters & trailers
✅ Deploy on a dedicated server for better performance

---

## 🤝 Contributing
Contributions are welcome! Feel free to open issues or submit pull requests to improve the project.

---

## 📜 License
This project is licensed under the Apache-2.0 License.

---

## 👨‍💻 Author
**Saket** - [GitHub](https://github.com/saketjha34)

---

📢 **Try the live app now:** [FilmFusion](https://filmfusionai.streamlit.app/) 🚀