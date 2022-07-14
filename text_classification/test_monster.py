from cookie_monster import *
import pytest
import sys


pytestmark = [pytest.mark.unit]


def setup_module():
    if not os.path.exists("mnist"):
        os.system(
            "curl https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.bz2 --output mnist.bz2"
        )
        os.system("bzip2 -d mnist.bz2")

    if not os.path.exists("mnist.t"):
        os.system(
            "curl https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.t.bz2 --output mnist.t.bz2"
        )
        os.system("bzip2 -d mnist.t.bz2")

    with open("string.txt", "w") as f:
        f.write("This is a test sentence\n")
        f.write("To train the model on mlm tasks\n")


def test_new_dimension():
    model = CookieMonster(
        100000, hidden_dimension=2000, hidden_sparsity=0.1, mlflow_enabled=False
    )
    model.set_output_dimension(10)
    assert model.output_layer.get_dim() == 10
    model.set_output_dimension(8)
    assert model.output_layer.get_dim() == 8


def test_train():
    model = CookieMonster(
        784, hidden_dimension=2000, hidden_sparsity=0.1, mlflow_enabled=False
    )
    model.eat_corpus("test_dir", False, True)
    model.evaluate("test_dir")


test_train()
