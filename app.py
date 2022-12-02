import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
import nltk

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
# Create flask app
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import re

from model import sentiment_predictor

flask_app = Flask(__name__)
flask_app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
flask_app.config['DEBUG'] = True

model = pickle.load(open("model.pkl", "rb"))


@flask_app.route('/')
def home():
    return render_template('index.html')


@flask_app.route("/predict", methods=['GET'])
def get_prediction():
    message = request.args.get('message')
    data = [message]
    prediction = sentiment_predictor(data)
    if prediction == 0:

        data = {
            "result": "negatif",

        }

        return jsonify(data)

    elif prediction == 1:
        data = {
            "result": "positif",

        }

        return jsonify(data)
    else:
        data = {
            "result": "rien",

        }

        return jsonify(data)




@flask_app.route('/predict', methods=['POST'])
def predict():

    message = request.args.get('message')
    data = [message]

    prediction = sentiment_predictor(data)

    if prediction == 0:
       return jsonify("negatif")

    elif prediction == 1:
        return jsonify("positif")
    else:
        return jsonify("rien")

if __name__ == "__main__":
    flask_app.run(debug=True)
