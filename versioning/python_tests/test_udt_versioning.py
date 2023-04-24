import os
from pathlib import Path

import pytest
from thirdai import bolt

SERIALIZED_CLASS_DIR = Path(__file__).resolve().parent / "serialized_classes"
ERROR_STRING = r"UDT_BASE, but got version 0"

pytestmark = [pytest.mark.unit]


def build_udt_model():
    # Dummy UDT model to test serialization

    model = bolt.UniversalDeepTransformer(
        data_types={
            "sample": bolt.types.text(),
            "target": bolt.types.categorical(),
        },
        target="target",
        n_target_classes=10,
    )

    return model


def test_save_load_udt_model():
    model = build_udt_model()
    model_name = "udt_model"
    model_path = os.path.join(SERIALIZED_CLASS_DIR, model_name)
    model.save(model_path)
    bolt.UniversalDeepTransformer.load(model_path)


def test_load_old_udt_model():
    model_name = "old_udt_model"
    model_path = os.path.join(SERIALIZED_CLASS_DIR, model_name)

    # Expected to raise error because the UDT model being loaded was saved
    # with version 0, which is older than any current udt version
    with pytest.raises(ValueError, match=ERROR_STRING):
        bolt.UniversalDeepTransformer.load(model_path)
