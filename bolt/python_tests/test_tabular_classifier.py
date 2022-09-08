from thirdai import bolt
import pytest
import os
import pandas as pd
from utils import remove_files, compute_accuracy_of_predictions

pytestmark = [pytest.mark.integration, pytest.mark.release]

CENSUS_INCOME_BASE_DOWNLOAD_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/"
)

TRAIN_FILE = "./census_income_train.csv"
TEST_FILE = "./census_income_test.csv"
PREDICTION_FILE = "./census_income_predictions.txt"

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


def setup_module():
    if not os.path.exists(TRAIN_FILE):
        os.system(
            f"curl {CENSUS_INCOME_BASE_DOWNLOAD_URL}adult.data --output {TRAIN_FILE}"
        )

    if not os.path.exists(TEST_FILE):
        os.system(
            f"curl {CENSUS_INCOME_BASE_DOWNLOAD_URL}adult.test --output {TEST_FILE}"
        )
        # reformat the test file
        with open(TEST_FILE, "r") as file:
            data = file.read().splitlines(True)
        with open(TEST_FILE, "w") as file:
            # for some reason each of the labels end with a "." in the test set
            # loop through data[1:] since the first line is bogus
            file.writelines([line.replace(".", "") for line in data[1:]])


def teardown_module():
    remove_files([TRAIN_FILE, TEST_FILE])


def get_census_income_metadata():
    df = pd.read_csv(TEST_FILE)
    n_classes = df[df.columns[-1]].nunique()
    column_datatypes = []
    for col_type in df.dtypes[:-1]:
        if col_type == "int64":
            column_datatypes.append("numeric")
        elif col_type == "object":
            column_datatypes.append("categorical")
    column_datatypes.append("label")

    test_labels = list(df[df.columns[-1]])

    return n_classes, column_datatypes, test_labels


def test_tabular_classifier_census_income_dataset():
    (n_classes, column_datatypes, test_labels) = get_census_income_metadata()
    classifier = bolt.TabularClassifier(
        hidden_layer_dim=1000, n_classes=n_classes, column_datatypes=column_datatypes
    )

    classifier.train(
        filename=TRAIN_FILE,
        epochs=1,
        learning_rate=0.01,
    )

    _, predictions = classifier.evaluate(filename=TEST_FILE)

    acc = compute_accuracy_of_predictions(test_labels, predictions)

    print("Computed Accuracy: ", acc)
    assert acc > 0.77


def create_single_test_samples():
    with open(TEST_FILE, "r") as file:
        lines = file.readlines()

        samples = []
        # skip the header
        for line in lines[1:]:
            # ignore the label column
            values = line.split(",")[:-1]
            samples.append(values)

    return samples


def test_tabular_classifier_predict_single():
    (n_classes, column_datatypes, _) = get_census_income_metadata()
    classifier = bolt.TabularClassifier(
        hidden_layer_dim=1000, n_classes=n_classes, column_datatypes=column_datatypes
    )
    
    classifier.train(
        filename=TRAIN_FILE,
        epochs=1,
        learning_rate=0.01,
    )

    _, predictions = classifier.evaluate(filename=TEST_FILE)

    single_test_samples = create_single_test_samples()

    for sample, expected_prediction in zip(single_test_samples, predictions):
        actual_prediction = classifier.predict(sample)
        assert actual_prediction == expected_prediction
