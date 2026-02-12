# =========================================================
# CUSTOMER SEGMENTATION TRAINING
# =========================================================

import pandas as pd
import pickle

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# =========================================================
# LOAD DATASET
# =========================================================

df = pd.read_csv("../datasets/customers.csv")

print("Dataset Loaded:", df.shape)

# =========================================================
# SELECT FEATURES
# =========================================================

X = df[[
    'Age',
    'Annual Income (k$)',
    'Spending Score (1-100)'
]]

# =========================================================
# SCALING
# =========================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================================================
# KMEANS MODEL
# =========================================================

kmeans = KMeans(
    n_clusters=5,
    random_state=42
)

clusters = kmeans.fit_predict(X_scaled)

df["Cluster"] = clusters

# =========================================================
# SAVE MODEL
# =========================================================

pickle.dump(
    {
        "model": kmeans,
        "scaler": scaler
    },
    open("../models/segmentation_model.pkl", "wb")
)

print("✅ Segmentation model saved")
