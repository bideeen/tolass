#!/usr/local/bin/python -V
from flask import Flask, render_template, url_for, request, jsonify
from flask_socketio import SocketIO
import json
import pickle
import random
from flask.globals import request
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer



# Invoking the flask function
app = Flask(__name__)
@app.route('/')
def index():

    predict_response = []

    if request.method == "POST":

        lemmatizer = WordNetLemmatizer

        intents = json.loads(open('data/intents.json').read())

        words = pickle.load(open('data/words.pkl', 'rb'))
        classes = pickle.load(open('data/classes.pkl', 'rb'))

        # in your prediction file.
        with open('models/rf_model', 'rb') as f:
            dtc = pickle.load(f)

        def clean_up_sentence(sentence):
            sentence_words = nltk.word_tokenize(sentence)
            sentence_words = [lemmatizer.lemmatize(word, word.lower()) for word in sentence_words]
            return sentence_words

        def bag_of_words(sentence):
            sentence_words = clean_up_sentence(sentence)
            bag = [0] * len(words)
            for w in sentence_words:
                for i, word in enumerate(words):
                    if word == w:
                        bag[i] = 1
            return np.array(bag)

        def predict_class(sentence):
            bow = bag_of_words(sentence)
            res = dtc.predict(np.array([bow]))[0]
            error_threshold = 0.25
            results = [[i,r] for i, r in enumerate(res) if r > error_threshold]
            
            results.sort(key=lambda x: x[1], reverse=True)
            return_list = []
            for r in results:
                return_list.append({'intent':classes[r[0]], 'probability': str(r[1])})
            return return_list

        def get_response(intents_list, intents_json):
            tag = intents_list[0]['intent']
            list_of_intents = intents_json['intents']
            for i in list_of_intents:
                if i['tags'] == tag:
                    result = random.choice(i['responses'])
                    break
            return result



        try:
            response = str(request.form['user'])
            # print(response)
            ints = predict_class(response)
            predict_response = get_response(ints, intents)

        except IndexError:
            pass

    return render_template('index.html', prediction = predict_response)



    


if __name__ == "__main__":
    app.run(debug=True, port=5991)