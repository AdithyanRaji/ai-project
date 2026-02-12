import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load dataset - try CSV first, then XLSX
try:
    df = pd.read_csv('../datasets/sentiment.csv')
except FileNotFoundError:
    df = pd.read_excel('../datasets/sentiment.xlsx')

X = df['text']
y = df['sentiment']

# Vectorization
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# Train model
model = LogisticRegression()
model.fit(X_vec, y)

# Save PKL
pickle.dump({
    "model": model,
    "vectorizer": vectorizer
}, open('../models/sentiment_model.pkl', 'wb'))

print("✅ Sentiment model trained and saved successfully!")

