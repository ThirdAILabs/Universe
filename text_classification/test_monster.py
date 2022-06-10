from http.cookiejar import Cookie
from cookie_monster import *
import pytest

def test_new_dimension():
    layers = [bolt.FullyConnected(dim=1000, sparsity=0.1, activation_function=bolt.ActivationFunctions.ReLU),
              bolt.FullyConnected(dim=2,  activation_function=bolt.ActivationFunctions.Softmax),]
    cookie_model = CookieMonster(layers)
    train_file = "/share/data/hwu/train_shuffled.csv"
    cookie_model.bolt_classifier.train(train_file=train_file, epochs=2, learning_rate=0.01)

    cookie_model.set_output_dimension(40)


def test_train():
    cookie_model = CookieMonster(output_dimension=64)
    cookie_model.train_corpus("/home/henry/cookie_train/")

test_new_dimension()
# test_train()