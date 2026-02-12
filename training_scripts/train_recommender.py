# =========================================================
# MOVIE RECOMMENDER TRAINING
# =========================================================

import pandas as pd
import pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================================================
# LOAD DATASET
# =========================================================

movies = pd.read_csv("../datasets/movies/movies.csv")

print("Movies loaded:", movies.shape)

# SAMPLE the first 5000 movies to avoid memory issues with large dataset
movies = movies.head(5000)
print(f"Using sample: {movies.shape}")

# =========================================================
# PREPROCESS GENRES
# =========================================================

# Replace | with space
movies["genres"] = movies["genres"].str.replace("|", " ")

# Vectorize genres
cv = CountVectorizer()

genre_matrix = cv.fit_transform(movies["genres"])

# =========================================================
# SIMILARITY MATRIX
# =========================================================

similarity = cosine_similarity(genre_matrix)

print("Similarity matrix created:", similarity.shape)

# =========================================================
# SAVE PKL
# =========================================================

pickle.dump(
    {
        "similarity": similarity,
        "movies": movies
    },
    open("../models/recommender.pkl", "wb")
)

print("✅ Recommender model saved (5000 movie sample)")
