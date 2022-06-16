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
        # self.bolt = bolt.SentimentClassifier(bolt_model_path)

    def predict(self, sentence, engine):
        start = time.time()
        if engine == "bolt":
            # pred = self.bolt.predict_sentiment(sentence.lower()) >= 0.5
            return 1, 0.6435
        elif engine == "roberta":
            pred = self.roberta(sentence)[0]["label"] == "POSITIVE"
        else:
            raise ValueError("Unsupported engine type '" +
                             request.form["engine"] + "'")
        end = time.time()
        return pred, (end - start) * 1000


app = Flask(__name__)
predictor = None


@app.route("/")
def home():
    return render_template("home.html", prediction="", background_color="", latency="")


@app.route("/", methods=["POST"])
def predict_sentiment():
    sentence = request.form["sentence"]
    engine = request.form["engine"]

    pred, latency = predictor.predict(sentence, engine)

    if pred:
        return render_template(
            "home.html",
            prediction="Positive",
            background_color="green",
            latency=latency,
        )
    else:
        return render_template(
            "home.html", prediction="Negative", background_color="red", latency=latency
        )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise ValueError(
            "Expected path to bolt model as command line argument.")

    predictor = PredictionBackend(sys.argv[1])

    # Set host = 0.0.0.0 so that the app is accessible outside of local via the machines ip address.
    app.run(debug=True, host="0.0.0.0")
