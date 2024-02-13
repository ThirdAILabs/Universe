import os
from pathlib import Path

import pytest
from thirdai import bolt

SERIALIZED_CLASS_DIR = Path(__file__).resolve().parent / "serialized_classes"
INTERNAL_ERROR_STRING = (
    r"Incompatible version. Expected version .* for UDT_BASE, but got version 0"
)
EXTERNAL_ERROR_STRING = r"The model you are loading is not compatible with the current version of thirdai \(.*\). Please downgrade to the version the model was saved with \(v0.0.0\)"


def build_udt_model():
    # Dummy UDT model to test serialization

    model = bolt.UniversalDeepTransformer(
        data_types={
            "input": bolt.types.text(),
            "output": bolt.types.categorical(),
        },
        target="output",
        n_target_classes=2,
        options={
            "input_dim": 10,
            "embedding_dimension": 10,
        },
    )

    return model


@pytest.mark.unit
def test_save_load_udt_model():
    model_name = "udt_model"
    model_path = os.path.join(SERIALIZED_CLASS_DIR, model_name)

    model = build_udt_model()
    model.save(model_path)
    bolt.UniversalDeepTransformer.load(model_path)


def create_udt_checkpoint():
    """
    This is the code used to generate the old udt checkpoint. It is not invoked
    in the test, but it is included here for future reference.
    """
    model = bolt.UniversalDeepTransformer(
        data_types={"text": bolt.types.text(), "id": bolt.types.categorical()},
        target="id",
        n_target_classes=100,
        options={"fhr": 1000, "embedding_dimension": 200},
    )

    model.train_batch([{"text": "some text", "id": "8"}])

    model.checkpoint("./simple_udt_checkpoint")
    model.save("./simple_udt_save")


@pytest.mark.unit
def test_load_old_udt_checkpoint():
    model_path = os.path.join(SERIALIZED_CLASS_DIR, "old_udt_checkpoint")

    bolt.UniversalDeepTransformer.load(model_path)


@pytest.mark.parametrize(
    "error_string",
    [
        pytest.param(INTERNAL_ERROR_STRING, marks=pytest.mark.unit),
        pytest.param(EXTERNAL_ERROR_STRING, marks=pytest.mark.release),
    ],
)
def test_load_old_udt_model(error_string):
    model_name = "old_udt_model"
    model_path = os.path.join(SERIALIZED_CLASS_DIR, model_name)

    # Expected to raise error because the UDT model being loaded was saved
    # with version 0, which is older than any current udt version
    with pytest.raises(ValueError, match=error_string):
        bolt.UniversalDeepTransformer.load(model_path)
