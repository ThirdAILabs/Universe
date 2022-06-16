from flask import Flask, request, render_template
from thirdai import bolt
import time
from transformers import pipeline
import torch

torch.set_num_threads(1)

app = Flask(__name__)

roberta_sentiment = pipeline("sentiment-analysis", model="siebert/sentiment-roberta-large-english")
bolt_sentiment = bolt.SentimentClassifier("../Universe/my_model")


def predict_with_bolt(sentence):
    pred = bolt_sentiment.predict_sentiment(sentence.lower())
    return pred >= 0.5


def predict_with_roberta(sentence):
    pred = roberta_sentiment(sentence)[0]
    return pred['label'] == "POSITIVE"


@app.route("/")
def home():
    return render_template("home.html", prediction="", background_color="", latency="")


@app.route("/", methods=['POST'])
def predict_sentiment():
    sentence = request.form["sentence"]
    start = time.time()
    if request.form["engine"] == "bolt":
        pred = predict_with_bolt(sentence)
    elif request.form["engine"] == "roberta":
        pred = predict_with_roberta(sentence)
    else:
        raise ValueError("Unsupported engine type '" + request.form["engine"] + "'")
    end = time.time()

    latency = 1000 * (end -start)
    if pred:
        return render_template("home.html", prediction="Positive", background_color="green", latency=latency)
    else:
        return render_template("home.html", prediction="Negative", background_color="red", latency=latency)

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0")

                                              
