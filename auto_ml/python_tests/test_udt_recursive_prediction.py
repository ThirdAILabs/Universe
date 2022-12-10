import os
import platform

import pytest
from thirdai import bolt


@pytest.fixture(scope="module")
def recursive_model():
    TRAIN_FILE = "./udt_recursive_data.csv"
    data = ["col,label,label_1,label_2,label_3\n", "1,1,2,3,4\n", "2,2,3,4,5\n"]

    with open(TRAIN_FILE, "w") as file:
        file.writelines(data)

    model = bolt.UniversalDeepTransformer(
        data_types={
            "col": bolt.types.categorical(),
            "label": bolt.types.categorical(),
            "label_1": bolt.types.categorical(),
            "label_2": bolt.types.categorical(),
            "label_3": bolt.types.categorical(),
        },
        target="label",
        n_target_classes=16,
        options={"prediction_depth": 4},
    )

    model.train(TRAIN_FILE, learning_rate=0.0001, epochs=1)

    os.remove(TRAIN_FILE)

    return model


def test_recursive_predict(recursive_model):
    predictions = recursive_model.predict({"col": "1"})
    assert predictions.shape == (4,)


def test_recursive_predict_batch(recursive_model):
    predictions = recursive_model.predict_batch([{"col": "1"}, {"col": "2"}])
    assert predictions.shape == (2, 4)


@pytest.mark.skipif(
    platform.system() == "Windows",
    reason="Throwing an exception leads to an access violation on windows.",
)
def test_error_without_expected_columns():
    with pytest.raises(
        ValueError,
        match="Expected column 'label_3' to be defined if prediction_depth=4.",
    ):
        bolt.UniversalDeepTransformer(
            data_types={
                "col": bolt.types.categorical(),
                "label": bolt.types.categorical(),
                "label_1": bolt.types.categorical(),
                "label_2": bolt.types.categorical(),
            },
            target="label",
            n_target_classes=16,
            options={"prediction_depth": 4},
        )


@pytest.mark.skipif(
    platform.system() == "Windows",
    reason="Throwing an exception leads to an access violation on windows.",
)
def test_error_if_column_has_wrong_type():
    with pytest.raises(
        ValueError, match="Expected column 'label_1' to be categorical."
    ):
        bolt.UniversalDeepTransformer(
            data_types={
                "col": bolt.types.categorical(),
                "label": bolt.types.categorical(),
                "label_1": bolt.types.numerical(range=(0, 1)),
                "label_2": bolt.types.categorical(),
                "label_3": bolt.types.categorical(),
            },
            target="label",
            n_target_classes=16,
            options={"prediction_depth": 4},
        )


@pytest.mark.skipif(
    platform.system() == "Windows",
    reason="Throwing an exception leads to an access violation on windows.",
)
def test_error_if_target_column_has_wrong_type():
    with pytest.raises(
        ValueError,
        match="Expected target column to be categorical if prediction_depth > 1 is used.",
    ):
        bolt.UniversalDeepTransformer(
            data_types={
                "col": bolt.types.categorical(),
                "label": bolt.types.numerical(range=(0, 1)),
                "label_1": bolt.types.categorical(),
                "label_2": bolt.types.categorical(),
                "label_3": bolt.types.categorical(),
            },
            target="label",
            n_target_classes=16,
            options={"prediction_depth": 4},
        )
