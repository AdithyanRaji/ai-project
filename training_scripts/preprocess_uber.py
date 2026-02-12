import pandas as pd

df = pd.read_csv("../datasets/uber.csv")

# Convert datetime
df['Date/Time'] = pd.to_datetime(df['Date/Time'])

# Extract features
df['hour'] = df['Date/Time'].dt.hour
df['day'] = df['Date/Time'].dt.day
df['month'] = df['Date/Time'].dt.month

# Save updated dataset
df.to_csv("../datasets/uber.csv", index=False)

print("✅ Uber dataset preprocessed")
