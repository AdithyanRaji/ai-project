from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
try:
    from models.drowsiness_detector import detect_drowsiness
except Exception:
    detect_drowsiness = None

app = Flask(__name__)

# =========================================================
# LOAD PKL FILES (Correct Structure)
# =========================================================

# Initialize variables so route functions don't raise NameError
sentiment_model = None
sentiment_vectorizer = None
fake_model = None
fake_vectorizer = None
parkinsons_model = None
parkinsons_scaler = None
fraud_model = None
fraud_scaler = None
similarity = None
movies_df = None

try:
    # Sentiment → model + vectorizer
    sentiment_data = pickle.load(open('models/sentiment_model.pkl', 'rb'))
    sentiment_model = sentiment_data["model"]
    sentiment_vectorizer = sentiment_data["vectorizer"]

    # Fake News → model + vectorizer
    fake_data = pickle.load(open('models/fake_news_model.pkl', 'rb'))
    fake_model = fake_data["model"]
    fake_vectorizer = fake_data["vectorizer"]

    # Parkinson → model (+ scaler optional)
    parkinsons_data = pickle.load(open('models/parkinsons_model.pkl', 'rb'))
    parkinsons_model = parkinsons_data["model"]
    parkinsons_scaler = parkinsons_data.get("scaler", None)

    # Fraud → model + scaler
    fraud_data = pickle.load(open('models/fraud_model.pkl', 'rb'))
    fraud_model = fraud_data["model"]
    fraud_scaler = fraud_data["scaler"]

    # Recommender → similarity + movies dataframe
    rec_data = pickle.load(open('models/recommender.pkl', 'rb'))
    similarity = rec_data["similarity"]
    movies_df = rec_data["movies"]
except FileNotFoundError as e:
    print(f"Error: Model file not found - {e}")
except Exception as e:
    print(f"Error loading models: {e}")

# =========================================================
@app.route('/')
def index():
    return render_template('index.html')

# =========================================================
# 1. SENTIMENT ANALYSIS
# =========================================================
@app.route('/sentiment', methods=['GET','POST'])
def sentiment():
    result = ""
    if request.method == 'POST':
        try:
            text = request.form.get('text', '')
            if not text:
                result = "Error: Please enter text"
            else:
                text_vec = sentiment_vectorizer.transform([text])
                prediction = sentiment_model.predict(text_vec)[0]
                result = "Positive" if prediction == 1 else "Negative"
        except Exception as e:
            result = f"Error: {str(e)}"

    return render_template('sentiment.html', result=result)

# =========================================================
# 2. FAKE NEWS DETECTION
# =========================================================
@app.route('/fake_news', methods=['GET','POST'])
def fake_news():
    result = ""
    if request.method == 'POST':
        try:
            news = request.form.get('news', '')
            if not news:
                result = "Error: Please enter news text"
            else:
                news_vec = fake_vectorizer.transform([news])
                prediction = fake_model.predict(news_vec)[0]
                result = "Real News" if prediction == 1 else "Fake News"
        except Exception as e:
            result = f"Error: {str(e)}"

    return render_template('fake_news.html', result=result)

# =========================================================
# 3. PARKINSONS DETECTION
# =========================================================
@app.route('/parkinsons', methods=['GET','POST'])
def parkinsons():
    result = ""
    if request.method == 'POST':
        try:
            features = [float(x) for x in request.form.values()]
            features = np.array(features).reshape(1, -1)

            if parkinsons_scaler:
                features = parkinsons_scaler.transform(features)

            prediction = parkinsons_model.predict(features)[0]
            result = "Parkinsons Detected" if prediction == 1 else "Healthy"
        except ValueError:
            result = "Error: Please enter valid numbers"
        except Exception as e:
            result = f"Error: {str(e)}"

    return render_template('parkinsons.html', result=result)

# =========================================================
# 4. SPEECH EMOTION
# =========================================================
@app.route('/speech_emotion', methods=['GET','POST'])
def speech_emotion():
    result = ""
    if request.method == 'POST':
        try:
            if 'audio' not in request.files:
                result = "Error: No audio file provided"
            else:
                file = request.files['audio']
                if file.filename == '':
                    result = "Error: No file selected"
                else:
                    result = "Speech emotion detection functionality coming soon"
        except Exception as e:
            result = f"Error: {str(e)}"
    
    return render_template('speech_emotion.html', result=result)


# =========================================================
# 6. DROWSINESS
# =========================================================
@app.route('/drowsiness', methods=['GET','POST'])
def drowsiness():

    result = ""

    if request.method == 'POST':
        try:
            detect_drowsiness()
            result = "Drowsiness detection started on webcam"

        except Exception as e:
            result = f"Error: {str(e)}"

    return render_template(
        'drowsiness.html',
        result=result
    )

# =========================================================
# 7. CREDIT CARD FRAUD
# =========================================================
@app.route('/fraud', methods=['GET','POST'])
def fraud():
    result = ""
    if request.method == 'POST':
        try:
            features = [float(x) for x in request.form.values()]
            features = np.array(features).reshape(1, -1)

            features = fraud_scaler.transform(features)
            prediction = fraud_model.predict(features)[0]

            result = "Fraudulent" if prediction == 1 else "Genuine"
        except ValueError:
            result = "Error: Please enter valid numbers"
        except Exception as e:
            result = f"Error: {str(e)}"

    return render_template('fraud.html', result=result)

# =========================================================
# 8. MOVIE RECOMMENDER
# =========================================================
@app.route('/recommender', methods=['GET','POST'])
def movie_recommender():
    movies = []

    if request.method == 'POST':
        try:
            name = request.form.get('movie', '')

            if not name:
                movies = []
            elif name in movies_df['title'].values:
                idx = movies_df[movies_df['title'] == name].index[0]
                scores = list(enumerate(similarity[idx]))
                scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:6]
                movies = [movies_df.iloc[i[0]].title for i in scores]
            else:
                movies = ["Movie not found"]
        except Exception as e:
            movies = [f"Error: {str(e)}"]

    return render_template('recommender.html', movies=movies)

# =========================================================
# 9. CUSTOMER SEGMENTATION
# =========================================================
@app.route('/segmentation')
def segmentation():
    try:
        df = pd.read_csv('datasets/customers.csv')
        return render_template(
            'customer_segmentation.html',
            tables=[df.to_html(classes='data')]
        )
    except FileNotFoundError:
        return render_template('customer_segmentation.html', tables=["<p>Dataset not found</p>"])
    except Exception as e:
        return render_template('customer_segmentation.html', tables=[f"<p>Error: {str(e)}</p>"])

# =========================================================
# 10. UBER DATA ANALYSIS
# =========================================================
@app.route('/uber')
def uber():
    try:
        df = pd.read_csv('datasets/uber.csv')
        trips = df['hour'].value_counts().sort_index()
        return render_template(
            'uber_analysis.html',
            trips=trips.to_dict()
        )
    except FileNotFoundError:
        return render_template('uber_analysis.html', trips={})
    except Exception as e:
        return render_template('uber_analysis.html', trips={})

# =========================================================
if __name__ == '__main__':
    app.run(debug=True)
