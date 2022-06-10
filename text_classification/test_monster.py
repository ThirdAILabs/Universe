from http.cookiejar import Cookie
from cookie_monster import *
import pytest


def test_new_dimension():
    layers = [
        bolt.FullyConnected(
            dim=1000, sparsity=0.1, activation_function=bolt.ActivationFunctions.ReLU
        ),
        bolt.FullyConnected(
            dim=64, activation_function=bolt.ActivationFunctions.Softmax
        ),
    ]
    cookie_model = CookieMonster(layers)
    train_file = "/share/data/hwu/train_shuffled.csv"
    cookie_model.bolt_classifier.train(
        train_file=train_file, epochs=2, learning_rate=0.01
    )

    cookie_model.set_output_dimension(40)


def test_train():
    layers = [
        bolt.FullyConnected(
            dim=1000, sparsity=0.1, activation_function=bolt.ActivationFunctions.ReLU
        ),
        bolt.FullyConnected(
            dim=64, activation_function=bolt.ActivationFunctions.Softmax
        ),
    ]
    cookie_model = CookieMonster(layers)
    cookie_model.train_corpus("/home/henry/cookie_train/", False)


def test_download():
    layers = [
        bolt.FullyConnected(
            dim=1000, sparsity=0.1, activation_function=bolt.ActivationFunctions.ReLU
        ),
        bolt.FullyConnected(
            dim=64, activation_function=bolt.ActivationFunctions.Softmax
        ),
    ]
    cookie_model = CookieMonster(layers)

    cookie_model.download_hidden_weights(
        "s3://mlflow-artifacts-199696198976/29/28efe03f75124dac9b234cdba38853ed/artifacts/weights_1000.npy",
        "s3://mlflow-artifacts-199696198976/29/28efe03f75124dac9b234cdba38853ed/artifacts/biases_1000.npy",
    )


test_train()
