from cookie_monster import *
import pytest

def test_new_dimension():
    model = CookieMonster(100000, hidden_dimension=2000, hidden_sparsity=0.1)
    model.set_output_dimension(10)
    assert model.output_layer.get_dim() == 10