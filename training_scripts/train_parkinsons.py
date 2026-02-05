import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv('../datasets/parkinsons.csv')

# Separate features and target
X = df.drop('status', axis=1)  # assuming 'status' is the target column
y = df['status']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# Save PKL with both model and scaler
pickle.dump({
    "model": model,
    "scaler": scaler
}, open('../models/parkinsons_model.pkl', 'wb'))

print("Parkinsons Model trained and saved successfully!")
