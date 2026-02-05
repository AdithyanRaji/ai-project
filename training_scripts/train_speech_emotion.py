# =========================================================
# SPEECH EMOTION RECOGNITION TRAINING
# =========================================================

import os
import numpy as np
import librosa
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# =========================================================
# DATASET PATH
# =========================================================

dataset_path = "../datasets/speech_emotion/"

# =========================================================
# EMOTION MAP
# =========================================================

emotion_map = {
    "01":"neutral",
    "02":"calm",
    "03":"happy",
    "04":"sad",
    "05":"angry",
    "06":"fearful",
    "07":"disgust",
    "08":"surprised"
}

# =========================================================
# FEATURE EXTRACTION FUNCTION
# =========================================================

def extract_features(file_path):

    audio, sample_rate = librosa.load(
        file_path,
        duration=3,
        offset=0.5
    )

    mfcc = np.mean(
        librosa.feature.mfcc(
            y=audio,
            sr=sample_rate,
            n_mfcc=40
        ).T,
        axis=0
    )

    return mfcc

# =========================================================
# LOAD DATASET
# =========================================================

features = []
labels = []

for root, dirs, files in os.walk(dataset_path):

    for file in files:

        if file.endswith(".wav"):

            file_path = os.path.join(root, file)

            # Extract emotion code
            emotion_code = file.split("-")[2]
            emotion = emotion_map[emotion_code]

            feature = extract_features(file_path)

            features.append(feature)
            labels.append(emotion)

print("✅ Dataset Loaded")

# =========================================================
# CONVERT TO NUMPY
# =========================================================

X = np.array(features)
y = np.array(labels)

# =========================================================
# ENCODE LABELS
# =========================================================

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2
)

# =========================================================
# BUILD NEURAL NETWORK
# =========================================================

model = Sequential()

model.add(Dense(256, input_shape=(40,), activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(len(set(y_encoded)), activation='softmax'))

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# =========================================================
# TRAIN MODEL
# =========================================================

print("🚀 Training Started...")

model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test)
)

print("✅ Training Completed")

# =========================================================
# SAVE MODEL + ENCODER
# =========================================================

model.save("../models/emotion_model.h5")

pickle.dump(
    encoder,
    open("../models/emotion_encoder.pkl","wb")
)

print("✅ Model Saved in /models")
