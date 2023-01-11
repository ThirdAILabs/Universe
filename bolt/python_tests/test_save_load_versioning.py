from thirdai import bolt
from pathlib import Path
import os
import pytest


pytestmark = [pytest.mark.unit, pytest.mark.release]

def save_model(path):
    input_layer = bolt.nn.Input(dim=1)
    output_layer = bolt.nn.FullyConnected(dim=1, activation="softmax")(input_layer)

    model = bolt.nn.Model(inputs=[input_layer], output=output_layer)
    model.compile(bolt.nn.losses.CategoricalCrossEntropy())

    model.save(path)


def load_model(path):
    bolt.nn.Model.load(path)


DIR_PATH = Path(__file__).resolve().parent
UNVERSIONED_MODEL_PATH = str(DIR_PATH / "unversioned_model.bolt")
INCORRECT_VERSION_MODEL_PATH = str(DIR_PATH / "incorrect_version_model.bolt")
CORRECT_VERSION_MODEL_PATH = "./correct_version_model.bolt"


def setup():
    save_model(CORRECT_VERSION_MODEL_PATH)


def teardown():
    os.remove(CORRECT_VERSION_MODEL_PATH)


def test_unversioned_model_load():
    with pytest.raises(RuntimeError):
        load_model(UNVERSIONED_MODEL_PATH)


def test_incorrect_version_model_load():
    with pytest.raises(RuntimeError):
        load_model(INCORRECT_VERSION_MODEL_PATH)


def test_correct_version_model_load():
    load_model(CORRECT_VERSION_MODEL_PATH)