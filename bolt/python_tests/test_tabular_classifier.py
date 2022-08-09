from thirdai import bolt
import pytest
import os
import pandas as pd
from .utils import remove_files, compute_accuracy_with_file

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
        with open(TRAIN_FILE, "r") as file:
            data = file.readlines()
        with open(TRAIN_FILE, "w") as file:
            # write column names as the header
            file.write(",".join(COLUMN_NAMES) + "\n")
            file.writelines([line for line in data])

    if not os.path.exists(TEST_FILE):
        os.system(
            f"curl {CENSUS_INCOME_BASE_DOWNLOAD_URL}adult.test --output {TEST_FILE}"
        )
        with open(TEST_FILE, "r") as fin:
            data = fin.read().splitlines(True)
        with open(TEST_FILE, "w") as fout:
            # write column names as the header
            fout.write(",".join(COLUMN_NAMES) + "\n")
            # for some reason each of the labels end with a "." in the test set
            # loop through data[1:] since the first line is bogus
            fout.writelines([line.replace(".", "") for line in data[1:]])


def teardown_module():
    remove_files([TRAIN_FILE, TEST_FILE, PREDICTION_FILE])


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
    classifier = bolt.TabularClassifier(model_size="medium", n_classes=n_classes)

    classifier.train(
        train_file=TRAIN_FILE,
        column_datatypes=column_datatypes,
        epochs=1,
        learning_rate=0.01,
    )

    classifier.predict(test_file=TEST_FILE, output_file=PREDICTION_FILE)

    acc = compute_accuracy_with_file(test_labels, PREDICTION_FILE)

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
    classifier = bolt.TabularClassifier(model_size="medium", n_classes=n_classes)

    classifier.train(
        train_file=TRAIN_FILE,
        column_datatypes=column_datatypes,
        epochs=1,
        learning_rate=0.01,
    )

    classifier.predict(test_file=TEST_FILE, output_file=PREDICTION_FILE)

    with open(PREDICTION_FILE) as pred:
        # remove trailing newline from each prediction
        expected_predictions = [x[:-1] for x in pred.readlines()]

    single_test_samples = create_single_test_samples()

    for sample, expected_prediction in zip(single_test_samples, expected_predictions):
        actual_prediction = classifier.predict_single(sample)
        assert actual_prediction == expected_prediction
