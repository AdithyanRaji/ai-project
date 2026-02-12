from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
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


@app.route('/sentiment', methods=['GET', 'POST'])
def sentiment():
    result = ""
    if request.method == 'POST':
        try:
            if sentiment_model is None or sentiment_vectorizer is None:
                result = "Error: Sentiment model not loaded"
            else:
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


@app.route('/fake_news', methods=['GET', 'POST'])
def fake_news():
    result = ""
    if request.method == 'POST':
        try:
            if fake_model is None or fake_vectorizer is None:
                result = "Error: Fake news model not loaded"
            else:
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


@app.route('/parkinsons', methods=['GET', 'POST'])
def parkinsons():
    result = ""
    if request.method == 'POST':
        try:
            if parkinsons_model is None:
                result = "Error: Parkinsons model not loaded"
            else:
                form_values = [v for k, v in request.form.items() if k.startswith('feature')]
                if not form_values:
                    form_values = [float(x) for x in request.form.values()]
                features = np.array([float(x) for x in form_values]).reshape(1, -1)

                if parkinsons_scaler:
                    features = parkinsons_scaler.transform(features)

                prediction = parkinsons_model.predict(features)[0]
                result = "Parkinsons Detected" if prediction == 1 else "Healthy"
        except ValueError:
            result = "Error: Please enter valid numbers only"
        except Exception as e:
            result = f"Error: {str(e)}"

    return render_template('parkinsons.html', result=result)

# =========================================================
# 4. SPEECH EMOTION
# =========================================================


@app.route('/speech_emotion', methods=['GET', 'POST'])
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
# 5. GENDER & AGE DETECTION
# =========================================================
@app.route('/gender_age', methods=['GET', 'POST'])
def gender_age():
    result = ""
    if request.method == 'POST':
        try:
            if 'image' not in request.files:
                result = "Error: No image file provided"
            else:
                file = request.files['image']
                if file.filename == '':
                    result = "Error: No file selected"
                else:
                    result = "Gender & age detection functionality coming soon"
        except Exception as e:
            result = f"Error: {str(e)}"

    return render_template('gender_age.html', result=result)


# =========================================================
# 6. DROWSINESS
# =========================================================
@app.route('/drowsiness', methods=['GET', 'POST'])
def drowsiness():
    result = ""
    if request.method == 'POST':
        try:
            if detect_drowsiness is None:
                result = "Error: Drowsiness detector not available (dlib/cv2 missing)"
            else:
                result = "Drowsiness detection functionality coming soon (requires desktop/webcam)"
        except Exception as e:
            result = f"Error: {str(e)}"
    return render_template('drowsiness.html', result=result)

# =========================================================
# 7. CREDIT CARD FRAUD
# =========================================================


@app.route('/fraud', methods=['GET', 'POST'])
def fraud():
    result = ""
    if request.method == 'POST':
        try:
            if fraud_model is None or fraud_scaler is None:
                result = "Error: Fraud model not loaded"
            else:
                form_values = [v for k, v in request.form.items() if k.startswith('feature')]
                if not form_values:
                    form_values = [float(x) for x in request.form.values()]
                features = np.array([float(x) for x in form_values]).reshape(1, -1)

                features = fraud_scaler.transform(features)
                prediction = fraud_model.predict(features)[0]

                result = "Fraudulent" if prediction == 1 else "Genuine"
        except ValueError:
            result = "Error: Please enter valid numbers only"
        except Exception as e:
            result = f"Error: {str(e)}"

    return render_template('fraud.html', result=result)

# =========================================================
# 8. MOVIE RECOMMENDER
# =========================================================
@app.route('/recommender', methods=['GET', 'POST'])
def movie_recommender():

    movies = []

    if request.method == 'POST':

        try:
            name = request.form.get('movie', '').strip()

            # Check empty input
            if not name:
                movies = ["Please enter a movie name"]

            # Check dataset loaded
            elif movies_df is None or similarity is None:
                movies = ["Recommender model not loaded"]

            else:

                # Case-insensitive partial match
                matches = movies_df[
                    movies_df['title']
                    .str.lower()
                    .str.contains(name.lower())
                ]

                if not matches.empty:

                    idx = matches.index[0]

                    scores = list(enumerate(similarity[idx]))
                    scores = sorted(
                        scores,
                        key=lambda x: x[1],
                        reverse=True
                    )[1:6]

                    movies = [
                        movies_df.iloc[i[0]].title
                        for i in scores
                    ]

                else:
                    movies = ["Movie not found in database"]

        except Exception as e:
            movies = [f"Error: {str(e)}"]

    return render_template(
        'recommender.html',
        movies=movies
    )

# =========================================================
# 9. CUSTOMER SEGMENTATION
# =========================================================

@app.route('/segmentation')
def segmentation():
    try:
        # Try to load customer data - check for both names
        csv_file = 'datasets/customers.csv'
        if not os.path.exists(csv_file):
            csv_file = 'datasets/Mall_Customers.csv'
        
        if not os.path.exists(csv_file):
            return render_template(
                'customer_segmentation.html',
                tables=["<p>Error: Customer dataset not found (expected: datasets/customers.csv or datasets/Mall_Customers.csv)</p>"]
            )
        
        df = pd.read_csv(csv_file)

        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        X = df[['Age','Annual Income (k$)','Spending Score (1-100)']]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        kmeans = KMeans(n_clusters=5, random_state=42)
        df['Cluster'] = kmeans.fit_predict(X_scaled)

        return render_template(
            'customer_segmentation.html',
            tables=[df.to_html(classes='data')],
        )
    except KeyError as e:
        return render_template(
            'customer_segmentation.html',
            tables=[f"<p>Error: Missing column {e} in dataset</p>"]
        )
    except Exception as e:
        return render_template(
            'customer_segmentation.html',
            tables=[f"<p>Error: {str(e)}</p>"]
        )

# =========================================================
# 10. UBER DATA ANALYSIS
# =========================================================



@app.route('/uber')
def uber():
    try:
        if not os.path.exists('datasets/uber.csv'):
            return render_template('uber_analysis.html', trips={}, plot_url=None, error="Dataset not found")
        
        df = pd.read_csv('datasets/uber.csv')

        # Extract hour from Date/Time column if it exists
        if 'Date/Time' in df.columns:
            df['DateTime'] = pd.to_datetime(df['Date/Time'])
            df['hour'] = df['DateTime'].dt.hour
        elif 'hour' not in df.columns:
            return render_template('uber_analysis.html', trips={}, plot_url=None, error="Dataset missing 'hour' or 'Date/Time' column")

        # Trips per hour
        trips = df['hour'].value_counts().sort_index()

        # Plot
        plt.figure(figsize=(8, 4))
        trips.plot(kind='bar')

        plt.title("Uber Trips by Hour")
        plt.xlabel("Hour")
        plt.ylabel("Trips")

        os.makedirs("static/plots", exist_ok=True)

        plot_path = "static/plots/uber_hourly.png"
        plt.savefig(plot_path)
        plt.close()

        return render_template(
            'uber_analysis.html',
            trips=trips.to_dict(),
            plot_url=plot_path,
            error=None
        )
    except FileNotFoundError:
        return render_template('uber_analysis.html', trips={}, plot_url=None, error="Dataset file not found")
    except KeyError as e:
        return render_template('uber_analysis.html', trips={}, plot_url=None, error=f"Missing column: {e}")
    except Exception as e:
        return render_template('uber_analysis.html', trips={}, plot_url=None, error=f"Error: {str(e)}")



# =========================================================
if __name__ == '__main__':
    app.run(debug=True)
