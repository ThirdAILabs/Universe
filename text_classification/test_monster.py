from cookie_monster import *
import pytest


def test_new_dimension():
    model = CookieMonster(100000, hidden_dimension=2000, hidden_sparsity=0.1)
    model.set_output_dimension(10)
    model.set_output_dimension(4)
    model.set_output_dimension(10)
    model.set_output_dimension(8)


def test_train():
    model = CookieMonster(500000, hidden_dimension=2000, hidden_sparsity=0.1, mlflow_enabled=False)
    model.eat_corpus("/home/henry/cookie_train/", False, True)
    model.set_output_dimension(4)
    model.eat_corpus("/home/henry/cookie_train/", False, True)

