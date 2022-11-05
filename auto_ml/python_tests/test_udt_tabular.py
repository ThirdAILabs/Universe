import os

import pytest
from model_test_utils import (
    compute_model_accuracy,
    compute_predict_accuracy,
    compute_predict_batch_accuracy,
    compute_saved_and_retrained_accuarcy,
)
from thirdai import bolt

pytestmark = [pytest.mark.unit, pytest.mark.release]

ACCURACY_THRESHOLD = 0.8


@pytest.fixture(scope="module")
def download_census_income():
    CENSUS_INCOME_BASE_DOWNLOAD_URL = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/"
    )
    TRAIN_FILE = "./census_income_train.csv"
    TEST_FILE = "./census_income_test.csv"
    COLUMN_NAMES = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "label",
    ]
    if not os.path.exists(TRAIN_FILE):
        os.system(
            f"curl {CENSUS_INCOME_BASE_DOWNLOAD_URL}adult.data --output {TRAIN_FILE}"
        )
        # reformat the train file
        with open(TRAIN_FILE, "r") as file:
            data = file.read().splitlines(True)
        with open(TRAIN_FILE, "w") as file:
            # Write header
            file.write(",".join(COLUMN_NAMES) + "\n")
            # Convert ", " delimiters to ",".
            file.writelines([line.replace(", ", ",") for line in data[1:]])

    if not os.path.exists(TEST_FILE):
        os.system(
            f"curl {CENSUS_INCOME_BASE_DOWNLOAD_URL}adult.test --output {TEST_FILE}"
        )
        # reformat the test file
        with open(TEST_FILE, "r") as file:
            data = file.read().splitlines(True)
        with open(TEST_FILE, "w") as file:
            # Write header
            file.write(",".join(COLUMN_NAMES) + "\n")
            # Convert ", " delimiters to ",".
            # Additionally, for some reason each of the labels end with a "." in the test set
            # loop through data[1:] since the first line is bogus
            file.writelines(
                [line.replace(".", "").replace(", ", ",") for line in data[1:]]
            )

    inference_samples = []
    with open(TEST_FILE, "r") as test_file:
        for line in test_file.readlines()[1:-1]:
            column_vals = {
                col_name: value
                for col_name, value in zip(COLUMN_NAMES, line.split(","))
            }
            label = column_vals["label"].strip()
            del column_vals["label"]
            inference_samples.append((column_vals, label))

    return TRAIN_FILE, TEST_FILE, inference_samples


@pytest.fixture(scope="module")
def train_udt_tabular(download_census_income):
    model = bolt.UniversalDeepTransformer(
        data_types={
            "age": bolt.types.numerical(range=(17, 90)),
            "workclass": bolt.types.categorical(n_unique_classes=9),
            "fnlwgt": bolt.types.numerical(range=(12285, 1484705)),
            "education": bolt.types.categorical(n_unique_classes=16),
            "education-num": bolt.types.categorical(n_unique_classes=16),
            "marital-status": bolt.types.categorical(n_unique_classes=7),
            "occupation": bolt.types.categorical(n_unique_classes=15),
            "relationship": bolt.types.categorical(n_unique_classes=6),
            "race": bolt.types.categorical(n_unique_classes=5),
            "sex": bolt.types.categorical(n_unique_classes=2),
            "capital-gain": bolt.types.numerical(range=(0, 99999)),
            "capital-loss": bolt.types.numerical(range=(0, 4356)),
            "hours-per-week": bolt.types.numerical(range=(1, 99)),
            "native-country": bolt.types.categorical(n_unique_classes=42),
            "label": bolt.types.categorical(n_unique_classes=2),
        },
        target="label",
    )

    train_filename, _, _ = download_census_income

    train_config = bolt.TrainConfig(epochs=5, learning_rate=0.01)
    model.train(train_filename, train_config)

    return model


def test_utd_tabular_accuracy(train_udt_tabular, download_census_income):
    model = train_udt_tabular
    _, test_filename, inference_samples = download_census_income

    acc = compute_model_accuracy(
        model, test_filename, inference_samples, use_class_name=True
    )
    assert acc >= ACCURACY_THRESHOLD


def test_udt_tabular_save_load(train_udt_tabular, download_census_income):
    model = train_udt_tabular
    train_filename, test_filename, inference_samples = download_census_income

    acc = compute_saved_and_retrained_accuarcy(
        model, train_filename, test_filename, inference_samples, use_class_name=True
    )
    assert acc >= ACCURACY_THRESHOLD


def test_udt_tabular_predict_single(train_udt_tabular, download_census_income):
    model = train_udt_tabular
    _, _, inference_samples = download_census_income

    acc = compute_predict_accuracy(model, inference_samples, use_class_name=True)
    assert acc >= ACCURACY_THRESHOLD


def test_udt_tabular_predict_batch(train_udt_tabular, download_census_income):
    model = train_udt_tabular
    _, _, inference_samples = download_census_income

    acc = compute_predict_batch_accuracy(model, inference_samples, use_class_name=True)
    assert acc >= ACCURACY_THRESHOLD
