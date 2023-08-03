import joblib
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.util import pr
from nltk.stem import SnowballStemmer
import numpy as np
from flask import Flask, render_template, request, jsonify

# Load the trained model and CountVectorizer
cv = joblib.load('count_vectorizer.joblib')
clf = joblib.load('savemodel.joblib')

# Preprocess text function
def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text = " ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text = " ".join(text)
    return text

app = Flask(__name__)

# Initialize NLTK and stopwords
nltk.download('stopwords')
stemmer = SnowballStemmer("english")
stopword = set(stopwords.words("english"))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        test_data = request.form['text']
        test_data_cleaned = clean(test_data)
        test_data_transformed = cv.transform([test_data_cleaned]).toarray()
        prediction = clf.predict(test_data_transformed)[0]
        if prediction == "nan":
            prediction = "No offensive words used"
        return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)

