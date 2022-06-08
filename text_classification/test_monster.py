from http.cookiejar import Cookie
from cookie_monster import *
import pytest

def test_new_dimension():
    cookie_model = CookieMonster(output_dimension=64)
    train_file = "/share/data/hwu/train_shuffled.csv"
    cookie_model.bolt_classifier.train(train_file=train_file, epochs=2, learning_rate=0.01)

    cookie_model.set_output_dimension(40)


test_new_dimension()