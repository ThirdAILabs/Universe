import os
import platform

import pytest
from thirdai import bolt

TRAIN_FILE = "./udt_recursive_data.csv"


@pytest.fixture(scope="module")
def recursive_model():
    data = ["col,label\n", "1,1 2 3 4\n", "2,2 3 4 5\n"]

    with open(TRAIN_FILE, "w") as file:
        file.writelines(data)

    model = bolt.UniversalDeepTransformer(
        data_types={
            "col": bolt.types.categorical(),
            "label": bolt.types.sequence(length=4, delimiter=" "),
        },
        target="label",
        n_target_classes=16,
    )

    model.train(TRAIN_FILE, learning_rate=0.0001, epochs=1)

    yield model

    os.remove(TRAIN_FILE)


def test_recursive_predict(recursive_model):
    predictions = recursive_model.predict({"col": "1"})
    assert predictions.shape == (4,)


def test_recursive_predict_batch(recursive_model):
    predictions = recursive_model.predict_batch([{"col": "1"}, {"col": "2"}])
    assert predictions.shape == (2, 4)


def test_recursive_evaluate(recursive_model):
    activations = recursive_model.evaluate(TRAIN_FILE)

    assert activations.shape == (8, 16)
