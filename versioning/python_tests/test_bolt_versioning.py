import os
from pathlib import Path

import pytest
from thirdai import bolt

SERIALIZED_CLASS_DIR = Path(__file__).resolve().parent / "serialized_classes"
INTERNAL_ERROR_STRING = (
    r"Incompatible version. Expected version .* for BOLT_MODEL, but got version 0"
)
EXTERNAL_ERROR_STRING = r"The model you are loading is not compatible with the current version of thirdai \(.*\). Please downgrade to the version the model was saved with \(v0.0.0\)"


def build_bolt_model():
    # Dummy bolt model to test serialization

    op = bolt.nn.FullyConnected(
        dim=2,
        input_dim=2,
        sparsity=0.4,
        activation="relu",
    )

    input_layer = bolt.nn.Input(dim=2)

    output_layer = op(input_layer)

    labels = bolt.nn.Input(dim=2)

    loss = bolt.nn.losses.CategoricalCrossEntropy(
        activations=output_layer, labels=labels
    )

    model = bolt.nn.Model(inputs=[input_layer], outputs=[output_layer], losses=[loss])

    return model


@pytest.mark.unit
def test_save_load_bolt_model():
    model_name = "bolt_model"
    model_path = os.path.join(SERIALIZED_CLASS_DIR, model_name)

    model = build_bolt_model()
    model.save(model_path)
    bolt.nn.Model.load(model_path)


@pytest.mark.parametrize(
    "error_string",
    [
        pytest.param(INTERNAL_ERROR_STRING, marks=pytest.mark.unit),
        pytest.param(EXTERNAL_ERROR_STRING, marks=pytest.mark.unit),
    ],
)
def test_load_old_bolt_model(error_string):
    model_name = "old_bolt_model"
    model_path = os.path.join(SERIALIZED_CLASS_DIR, model_name)

    # Expected to raise error because the bolt model being loaded was saved
    # with version 0, which is older than any current bolt version
    with pytest.raises(ValueError, match=error_string):
        bolt.nn.Model.load(model_path)
