import os

import pytest
from cookie_monster import CookieMonster
from thirdai.dataset import FixedVocabulary

pytestmark = [pytest.mark.unit]

BERT_VOCAB_PATH = "bert-base-uncased.vocab"
BERT_VOCAB_URL = "https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt"


def setup_module():
    with open("string_mlm.txt", "w") as f:
        f.write("This is a test sentence\n")
        f.write("To train the model on mlm tasks\n")
    with open("string_classification.csv", "w") as f:
        f.write("0,This is a positive sentence\n")
        f.write("1,To train the model on classification tasks\n")

    if not os.path.exists(BERT_VOCAB_PATH):
        import urllib.request

        response = urllib.request.urlopen(BERT_VOCAB_URL)
        with open(BERT_VOCAB_PATH, "wb+") as bert_vocab_file:
            bert_vocab_file.write(response.read())


def test_new_dimension():
    vocab = FixedVocabulary(BERT_VOCAB_PATH)
    model = CookieMonster(
        vocab,
        input_dimension=100000,
        hidden_dimension=2000,
        hidden_sparsity=0.1,
        mlflow_enabled=False,
    )
    model.set_output_dimension(10)
    assert model.output_layer.get_dim() == 10
    model.set_output_dimension(30224)
    assert model.output_layer.get_dim() == 30224


def test_train():
    vocab = FixedVocabulary(BERT_VOCAB_PATH)
    model = CookieMonster(
        vocab,
        input_dimension=784,
        hidden_dimension=2000,
        hidden_sparsity=0.1,
        mlflow_enabled=False,
    )
    model.eat_corpus("test_dir", False, True)
    model.evaluate("test_dir")
