import json
import os
from pathlib import Path

import pytest
from thirdai import bolt, deployment

SERIALIZED_CLASS_DIR = Path(__file__).resolve().parent / "serialized_classes"
ERROR_STRING = r"UDT_BASE, but got version 0"

pytestmark = [pytest.mark.unit]


def build_udt_model():
    # Dummy UDT model to test serialization

    model_config = {
        "inputs": ["input"],
        "nodes": [
            {
                "name": "hidden",
                "type": "fully_connected",
                "dim": 10,
                "sparsity": 0.1,
                "activation": "relu",
                "predecessor": "input",
            },
            {
                "name": "output",
                "type": "fully_connected",
                "dim": {"param_name": "output_dim"},
                "sparsity": 0.1,
                "activation": "sigmoid",
                "predecessor": "hidden",
            },
        ],
        "output": "output",
        "loss": "BinaryCrossEntropyLoss",
    }
    model_config_path = os.path.join(SERIALIZED_CLASS_DIR, "udt_model_config")
    deployment.dump_config(
        config=json.dumps(model_config),
        filename=model_config_path,
    )

    model = bolt.UniversalDeepTransformer(
        data_types={
            "input": bolt.types.text(),
            "output": bolt.types.categorical(),
        },
        target="output",
        n_target_classes=2,
        model_config=model_config_path,
    )

    return model


def test_save_load_udt_model():
    model_name = "udt_model"
    model_path = os.path.join(SERIALIZED_CLASS_DIR, model_name)

    model = build_udt_model()
    model.save(model_path)
    bolt.UniversalDeepTransformer.load(model_path)


def test_load_old_udt_model():
    model_name = "old_udt_model"
    model_path = os.path.join(SERIALIZED_CLASS_DIR, model_name)

    # Expected to raise error because the UDT model being loaded was saved
    # with version 0, which is older than any current udt version
    with pytest.raises(ValueError, match=ERROR_STRING):
        bolt.UniversalDeepTransformer.load(model_path)
