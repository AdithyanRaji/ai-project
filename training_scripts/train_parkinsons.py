import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load dataset - try parkinsons.csv first, then parkinsons.data
try:
    df = pd.read_csv('../datasets/parkinsons.csv')
except FileNotFoundError:
    df = pd.read_csv('../datasets/parkinsons.data')

print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()[:5]}...")  # First 5 columns

# Drop non-numeric columns (like 'name' if it exists)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"Numeric columns: {len(numeric_cols)}")

# Separate features and target
# 'status' is the target column
if 'status' in df.columns:
    X = df.drop(['status'] + [c for c in df.columns if c not in numeric_cols and c != 'status'], axis=1)
    y = df['status']
else:
    raise ValueError("'status' column not found in parkinsons dataset")

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

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

print("✅ Parkinsons Model trained and saved successfully!")


