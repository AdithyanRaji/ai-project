import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv('../datasets/movies.csv')

# Example: using genre features
features = movies.drop(['title'], axis=1)

similarity = cosine_similarity(features)

pickle.dump({
    "similarity": similarity,
    "movies": movies
}, open('../models/recommender.pkl', 'wb'))
