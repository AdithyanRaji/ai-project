from flask import Flask, render_template, request
import pickle
import numpy as np


app = Flask(__name__)


# Load Models
sentiment_model = pickle.load(open('models/sentiment_model.pkl', 'rb'))
fake_model = pickle.load(open('models/fake_news_model.pkl', 'rb'))
parkinsons_model = pickle.load(open('models/parkinsons_model.pkl', 'rb'))
fraud_model = pickle.load(open('models/fraud_model.pkl', 'rb'))
recommender = pickle.load(open('models/recommender.pkl', 'rb'))


@app.route('/')
def index():
    return render_template('index.html')


# =========================================================
# 1. SENTIMENT ANALYSIS-/
# =========================================================
@app.route('/sentiment', methods=['GET','POST'])
def sentiment():
    result = ""
    if request.method == 'POST':
    text = request.form['text']
    prediction = sentiment_model.predict([text])[0]
    result = "Positive" if prediction == 1 else "Negative"
    return render_template('sentiment.html', result=result)

# =========================================================
# 2. FAKE NEWS DETECTION-/
# =========================================================
@app.route('/fake_news', methods=['GET','POST'])
def fake_news():
    result = ""
    if request.method == 'POST':
    news = request.form['news']
    prediction = fake_model.predict([news])[0]
    result = "Real News" if prediction == 1 else "Fake News"
    return render_template('fake_news.html', result=result)


# =========================================================
# 3. PARKINSONS DETECTION-/
# =========================================================
@app.route('/parkinsons', methods=['GET','POST'])
def parkinsons():
    result = ""
    if request.method == 'POST':
    features = [float(x) for x in request.form.values()]
    prediction = parkinsons_model.predict([features])[0]
    result = "Parkinsons Detected" if prediction == 1 else "Healthy"
    return render_template('parkinsons.html', result=result)


# =========================================================
# 4. SPEECH EMOTION RECOGNITION (Placeholder Inference)
# =========================================================
@app.route('/speech_emotion')
def speech_emotion():
    return render_template('speech_emotion.html')

# =========================================================
# 5. GENDER & AGE DETECTION
# =========================================================
@app.route('/gender_age')
def gender_age():
    return render_template('gender_age.html')


# =========================================================
# 6. DRIVER DROWSINESS
# =========================================================
@app.route('/drowsiness')
def drowsiness():
    return render_template('drowsiness.html')


# =========================================================
# 7. CREDIT CARD FRAUD-/
# =========================================================
@app.route('/fraud', methods=['GET','POST'])
def fraud():
    result = ""
    if request.method == 'POST':
    features = [float(x) for x in request.form.values()]
    prediction = fraud_model.predict([features])[0]
    result = "Fraudulent" if prediction == 1 else "Genuine"
    return render_template('fraud.html', result=result)


# =========================================================
# 8. MOVIE RECOMMENDER-/
# =========================================================
@app.route('/recommender', methods=['GET','POST'])
def movie_recommender():
    movies = []
    if request.method == 'POST':
    name = request.form['movie']
    movies = recommender.get(name, [])
    return render_template('reccommender.html', movies=movies)


# =========================================================
# 9. CUSTOMER SEGMENTATION (Visualization Placeholder)
# =========================================================
@app.route('/segmentation')
def segmentation():
    df=pd.read_csv('datasets/customers.csv')
    return render_template(
        'segmentation.html',
        tables=[df.to_html(classes='data')]
    )


# =========================================================
# 10. UBER DATA ANALYSIS
# =========================================================
@app.route('/uber')
def uber():
    df=pd.read_csv('datasets/uber.csv')
    trips=df['hour'].value_counts().sort_index()
    return render_template('uber.html',trips=trips.to_dict())





# =========================================================
if __name__ == '__main__':
app.run(debug=True)
