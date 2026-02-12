# AI Project Dashboard

A comprehensive Flask-based web application featuring 10 AI/ML capabilities including sentiment analysis, fake news detection, fraud detection, movie recommendations, customer segmentation, and more.

## 📋 Features

### 1. **Sentiment Analysis**
- Classify text as Positive or Negative
- Uses TF-IDF vectorization + Logistic Regression
- Route: `/sentiment`

### 2. **Fake News Detection**
- Detect if news articles are real or fake
- Uses TF-IDF vectorization + Naive Bayes classifier
- Route: `/fake_news`

### 3. **Parkinson's Disease Detection**
- Predict presence of Parkinson's based on voice features
- Uses Random Forest classifier with K-scaling
- Route: `/parkinsons`

### 4. **Speech Emotion Recognition**
- Recognize emotions from audio (alpha/beta)
- Route: `/speech_emotion`

### 5. **Gender & Age Detection**
- Detect gender and age from images (alpha/beta)
- Route: `/gender_age`

### 6. **Drowsiness Detection**
- Real-time webcam-based drowsiness detection
- Uses dlib face detection + eye aspect ratio (EAR)
- Route: `/drowsiness`

### 7. **Credit Card Fraud Detection**
- Classify transactions as fraudulent or genuine
- Uses Random Forest with StandardScaler
- Route: `/fraud`

### 8. **Movie Recommender System**
- Get movie recommendations based on similarity
- Uses cosine similarity on genre features
- Route: `/recommender`

### 9. **Customer Segmentation**
- K-Means clustering of customers by behavior
- Features: Age, Annual Income, Spending Score
- Route: `/segmentation`

### 10. **Uber Trip Analysis**
- Analyze Uber trips by hour with visualization
- Generates historical trend plots
- Route: `/uber`

---

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Setup

1. **Clone the repository**
```bash
git clone <repo-url>
cd ai-project
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download required data files**
   - Place pre-trained models in `models/` directory
   - Place datasets in `datasets/` directory
   - See [Dataset Requirements](#-dataset-requirements) section

---

## 📊 Dataset Requirements

| Feature | Required File | Format | Notes |
|---------|---------------|--------|-------|
| Sentiment | `datasets/sentiment.csv` | CSV | Must have 'text' and 'sentiment' columns |
| Fake News | `datasets/fake_news.csv` | CSV | Must have 'text' and 'label' columns |
| Parkinson's | `datasets/parkinsons.data` | Data file | Must have 'status' column (target) |
| Fraud | `datasets/fraud.csv` | CSV | Must have 'Class' column (target) |
| Customers | `datasets/customers.csv` OR `Mall_Customers.csv` | CSV | Must have Age, Annual Income, Spending Score |
| Uber | `datasets/uber.csv` | CSV | Must have 'hour' column |
| Movies | `datasets/movies/movies.csv` | CSV | Must have 'title' and 'genres' columns |
| Speech Emotion | `datasets/speech_emotion/` | Audio files | Directory-based structure |

---

## 🏃 Running the Application

### Development Mode
```bash
python app.py
```
- Server runs on `http://localhost:5000`
- Debug mode: ON (auto-reload on file changes)

### Production Mode
```bash
gunicorn app:app
```

---

## 📁 Project Structure

```
ai-project/
├── app.py                      # Main Flask application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── datasets/                   # Data files
│   ├── customers.csv
│   ├── fake_news.csv
│   ├── fraud.csv
│   ├── parkinsons.data
│   ├── sentiment.xlsx
│   ├── uber.csv
│   ├── movies/
│   └── speech_emotion/
├── models/                     # Pre-trained ML models (.pkl files)
│   ├── sentiment_model.pkl
│   ├── fake_news_model.pkl
│   ├── parkinsons_model.pkl
│   ├── fraud_model.pkl
│   ├── recommender.pkl
│   └── shape_predictor_68_face_landmarks.dat
├── templates/                  # HTML templates
│   ├── index.html
│   ├── sentiment.html
│   ├── fake_news.html
│   ├── parkinsons.html
│   ├── speech_emotion.html
│   ├── gender_age.html
│   ├── drowsiness.html
│   ├── fraud.html
│   ├── recommender.html
│   ├── customer_segmentation.html
│   └── uber_analysis.html
├── training_scripts/           # Model training scripts
│   ├── train_sentiment.py
│   ├── train_fake_news.py
│   ├── train_parkinsons.py
│   ├── train_fraud.py
│   ├── train_recommender.py
│   ├── train_segmentation.py
│   ├── train_speech_emotion.py
│   └── preprocess_uber.py
├── models/                     # Python modules
│   └── drowsiness_detector.py  # Drowsiness detection logic
└── static/                     # Generated plots
    └── plots/
        └── uber_hourly.png
```

---

## 🔧 Training Models

To retrain models from scratch:

```bash
cd training_scripts

# Train individual models
python train_sentiment.py
python train_fake_news.py
python train_parkinsons.py
python train_fraud.py
python train_recommender.py
python train_segmentation.py
python train_speech_emotion.py
```

**Note:** Models require respective datasets in `datasets/` folder.

---

## 📝 API Endpoints

| Route | Method | Description |
|-------|--------|-------------|
| `/` | GET | Home page dashboard |
| `/sentiment` | GET, POST | Sentiment analysis |
| `/fake_news` | GET, POST | Fake news detection |
| `/parkinsons` | GET, POST | Parkinson's detection |
| `/speech_emotion` | GET, POST | Speech emotion (upload audio) |
| `/gender_age` | GET, POST | Gender & age detection (upload image) |
| `/drowsiness` | GET, POST | Drowsiness detection |
| `/fraud` | GET, POST | Fraud detection |
| `/recommender` | GET, POST | Movie recommendations |
| `/segmentation` | GET | Customer segmentation analysis |
| `/uber` | GET | Uber trip analysis |

---

## ⚠️ Error Handling

The application includes comprehensive error handling:

- **Missing models**: Routes return user-friendly error messages (models not loaded)
- **Missing datasets**: Routes gracefully degrade with file-not-found messages
- **Invalid inputs**: Form validation with helpful error messages
- **Processing errors**: All exceptions caught and displayed to users

Example error responses:
- "Error: Sentiment model not loaded" (if model file corrupt/missing)
- "Error: Please enter valid numbers only" (if form input invalid for numeric routes)
- "Error: Dataset not found" (if CSV files missing)

---

## 🛠️ Dependencies

### Core
- Flask 3.0.0
- Pandas 2.2.0
- NumPy 1.26.4

### ML/Data Science
- scikit-learn 1.4.0
- TensorFlow 2.15.0 (optional)
- PyTorch 2.2.0 (optional)

### Computer Vision
- OpenCV 4.9.0.80
- dlib 19.24.2
- face-recognition 1.3.0

### Audio Processing
- librosa 0.10.1
- PyAudio 0.2.14

### Visualization
- Matplotlib 3.8.2
- Seaborn 0.13.2
- Plotly 5.18.0

See `requirements.txt` for full list with versions.

---

## 📋 Known Limitations

1. **Speech Emotion & Gender/Age**: Currently placeholders - full implementation requires model training
2. **Drowsiness Detection**: Requires desktop/webcam access (won't work in headless/server environments)
3. **Form Extraction**: Numeric routes assume form fields contain only numeric values
4. **Static Files**: Uber plot generation creates `static/plots/` directory on-demand

---

## 🔐 Security Notes

- Application runs with `debug=True` in development
- For production, use environment variables for secrets
- Validate all file uploads before processing
- Sanitize user inputs in production deployment

---

## 📞 Support

For issues or questions:
1. Check error messages in browser console
2. Review logs in terminal where Flask is running
3. Verify datasets are in correct location and format
4. Ensure all models are properly trained/loaded

---

## 📄 License

This project is open source. Add your license here.

---

## 🎯 Future Enhancements

- [ ] Implement gender/age detection with trained model
- [ ] Implement speech emotion recognition
- [ ] Add user authentication
- [ ] Create API endpoints for programmatic access
- [ ] Add data visualization dashboard
- [ ] Implement model versioning
- [ ] Add batch prediction capability
- [ ] Create Docker containerization

---

**Last Updated:** February 12, 2026