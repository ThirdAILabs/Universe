from flask import Flask, request, render_template
from thirdai import bolt
import time
from transformers import pipeline
import torch
import sys

torch.set_num_threads(1)


class PredictionBackend:
    def __init__(self, bolt_model_path):
        self.roberta = pipeline(
            "sentiment-analysis", model="siebert/sentiment-roberta-large-english"
        )
        self.bolt = bolt.SentimentClassifier(bolt_model_path)

    def predict(self, sentence):
        start = time.time()
        bolt_pred = self.bolt.predict_sentiment(sentence.lower()) >= 0.5
        end = time.time()
        bolt_latency = 1000 * (end - start)
        bolt_latency = round(bolt_latency, 4)

        start = time.time()
        roberta_pred = self.roberta(sentence)[0]["label"] == "POSITIVE"
        end = time.time()
        roberta_latency = 1000 * (end - start)
        roberta_latency = round(roberta_latency, 4)

        return bolt_pred, bolt_latency, roberta_pred, roberta_latency


app = Flask(__name__)
predictor = None


def get_color(pred):
    if pred:
        return "rgb(8, 110, 20)"
    return "rgb(161, 34, 19)"


def get_pred_name(pred):
    if pred:
        return "Positive"
    return "Negative"


@app.route("/")
def home():
    return render_template(
        "home.html",
    )


@app.route("/", methods=["POST"])
def predict_sentiment():
    sentence = request.form["query"]

    bolt_pred, bolt_latency, roberta_pred, roberta_latency = predictor.predict(
        sentence)

    return render_template(
        "home.html",
        bolt_background=get_color(bolt_pred),
        bolt_prediction=get_pred_name(bolt_pred),
        bolt_latency=bolt_latency,
        roberta_background=get_color(roberta_pred),
        roberta_prediction=get_pred_name(roberta_pred),
        roberta_latency=roberta_latency,
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise ValueError(
            "Expected path to bolt model as command line argument.")

    predictor = PredictionBackend(sys.argv[1])

    # Set host = 0.0.0.0 so that the app is accessible outside of local via the machines ip address.
    app.run(debug=False, host="0.0.0.0")
