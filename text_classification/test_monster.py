from cookie_monster import *
import pytest


def test_new_dimension():
    model = CookieMonster(100000, hidden_dimension=2000, hidden_sparsity=0.1)
    model.set_output_dimension(10)
    assert model.output_layer.get_dim() == 10


def test_train():
    model = CookieMonster(500000, hidden_dimension=2000, hidden_sparsity=0.1, mlflow_enabled=False)
    # model.train_corpus("/home/henry/cookie_train/", False, True)
    model.evaluate("/home/henry/cookie_test/")
    # model.evaluate("/home/henry/cookie_test/")

# test_new_dimension()
test_train()
