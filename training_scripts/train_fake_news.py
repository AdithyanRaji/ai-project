import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load dataset
df = pd.read_csv('../datasets/fake_news.csv')

X = df['text']
y = df['label']

# Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_vec = vectorizer.fit_transform(X)

# Train model
model = MultinomialNB()
model.fit(X_vec, y)

# Save PKL
pickle.dump({
    "model": model,
    "vectorizer": vectorizer
}, open('../models/fake_news_model.pkl', 'wb'))

print("Fake News Model trained and saved successfully!")
