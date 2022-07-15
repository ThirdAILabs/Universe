from thirdai import bolt
import pytest
import os
import pandas as pd
from utils import remove_files, compute_accuracy

CENSUS_INCOME_BASE_DOWNLOAD_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/"
)

TRAIN_FILE = "./census_income_train.csv"
TEST_FILE = "./census_income_test.csv"
PREDICTION_FILE = "./census_income_predictions.txt"


def download_census_income_dataset():
    if not os.path.exists(TRAIN_FILE):
        os.system(
            f"curl {CENSUS_INCOME_BASE_DOWNLOAD_URL}adult.data --output {TRAIN_FILE}"
        )
    if not os.path.exists(TEST_FILE):
        os.system(
            f"curl {CENSUS_INCOME_BASE_DOWNLOAD_URL}adult.test --output {TEST_FILE}"
        )

    with open(TEST_FILE, "r") as fin:
        data = fin.read().splitlines(True)
    with open(TEST_FILE, "w") as fout:
        # first line is bogus so delete it
        # for some reason each of the labels end with a "." in the test set
        fout.writelines([line.replace(".", "") for line in data[1:]])

    df = pd.read_csv(TEST_FILE)
    n_classes = df[df.columns[-1]].nunique()
    column_datatypes = []
    for col_type in df.dtypes[:-1]:
        if col_type == "int64":
            column_datatypes.append("numeric")
        elif col_type == "object":
            column_datatypes.append("categorical")
    column_datatypes.append("label")

    # theres no header so add the first column name as part of the labels
    test_labels = [df.columns[-1]] + list(df[df.columns[-1]])

    return n_classes, column_datatypes, test_labels


@pytest.mark.integration
@pytest.mark.release
def test_tabular_classifier_census_income_dataset():
    (n_classes, column_datatypes, test_labels) = download_census_income_dataset()
    classifier = bolt.TabularClassifier(model_size="medium", n_classes=n_classes)

    classifier.train(
        train_file=TRAIN_FILE,
        column_datatypes=column_datatypes,
        epochs=1,
        learning_rate=0.01,
    )

    classifier.predict(test_file=TEST_FILE, output_file=PREDICTION_FILE)

    acc = compute_accuracy(test_labels, PREDICTION_FILE)

    print("Computed Accuracy: ", acc)
    assert acc > 0.79

    remove_files([TRAIN_FILE, TEST_FILE, PREDICTION_FILE])
