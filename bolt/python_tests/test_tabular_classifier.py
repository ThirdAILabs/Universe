from multiprocessing.sharedctypes import Value
from thirdai import bolt
import pytest
import os
import pandas as pd
from utils import remove_files, compute_accuracy_of_predictions, check_autoclassifier_predict_correctness

pytestmark = [pytest.mark.integration, pytest.mark.release]

CENSUS_INCOME_BASE_DOWNLOAD_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/"
)

TRAIN_FILE = "./census_income_train.csv"
TEST_FILE = "./census_income_test.csv"
SAVE_FILE = "./temporary_tabular_classifier"

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
    """
    This test creates and trains a tabular classifier on the census income
    dataset and checks that it acheives the correct accuracy. Then it saves the
    trained classifier, reloads it and ensures that the results of predict match
    the predictions computed on the entire dataset.
    """
    (n_classes, column_datatypes, test_labels) = get_census_income_metadata()
    classifier = bolt.TabularClassifier(
        internal_model_dim=1000, n_classes=n_classes, column_datatypes=column_datatypes
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

    classifier.save(SAVE_FILE)

    new_classifier = bolt.TabularClassifier.load(SAVE_FILE)

    single_test_samples = create_single_test_samples()

    check_autoclassifier_predict_correctness(new_classifier, single_test_samples, predictions)


def create_single_test_samples():
    with open(TEST_FILE, "r") as file:
        lines = file.readlines()

        samples = []
        # skip the header and the last line since it is empty
        for line in lines[1:-1]:
            # ignore the label column
            values = line.split(",")[:-1]
            samples.append(values)

    return samples


TEMP_TABULAR_TRAIN_FILE = "./temp_tabular_classifier_train_file"


def create_temp_file(contents):
    with open(TEMP_TABULAR_TRAIN_FILE, "w") as file:
        file.writelines(contents)


def remove_temp_file():
    os.remove(TEMP_TABULAR_TRAIN_FILE)


def test_evaluate_before_train_throws():
    create_temp_file(["colname1,colname2\n", "value1,label1\n", "value3,label2\n"])

    column_datatypes = ["categorical", "label"]
    classifier = bolt.TabularClassifier(
        internal_model_dim=1, n_classes=1, column_datatypes=column_datatypes
    )

    with pytest.raises(
        RuntimeError,
        match=r"Cannot call evaulate on TabularClassifier before calling train.",
    ):
        classifier.evaluate(TEMP_TABULAR_TRAIN_FILE)

    with pytest.raises(
        RuntimeError,
        match=r"Cannot call featurizeInputForInference on TabularClasssifier before "
        "training.",
    ):
        classifier.predict(["cat1"])

    classifier.save(SAVE_FILE)
    classifier = bolt.TabularClassifier.load(SAVE_FILE)

    with pytest.raises(
        RuntimeError,
        match=r"Cannot call evaulate on TabularClassifier before calling train.",
    ):
        classifier.evaluate(TEMP_TABULAR_TRAIN_FILE)

    with pytest.raises(
        RuntimeError,
        match=r"Cannot call featurizeInputForInference on TabularClasssifier before "
        "training.",
    ):
        classifier.predict(["cat1"])

    remove_temp_file()


def test_column_datatypes_mismatch():
    create_temp_file(["colname1,colname2\n", "value1,label1\n", "value3,label2\n"])

    column_datatypes = ["label"]
    classifier = bolt.TabularClassifier(
        internal_model_dim=1, n_classes=1, column_datatypes=column_datatypes
    )

    with pytest.raises(ValueError, match=r"Csv format error. Expected*"):
        classifier.train(TEMP_TABULAR_TRAIN_FILE, epochs=1, learning_rate=0.1)

    remove_temp_file()


def test_train_evaluate_column_mismatch():
    create_temp_file(["colname1,colname2\n", "value1,label1\n", "value3,label2\n"])

    column_datatypes = ["categorical", "label"]
    classifier = bolt.TabularClassifier(
        internal_model_dim=1, n_classes=2, column_datatypes=column_datatypes
    )

    classifier.train(TEMP_TABULAR_TRAIN_FILE, epochs=1, learning_rate=0.1)

    create_temp_file(["colname1,colname2,colname3\n", "value1,value2,label1\n"])

    with pytest.raises(
        ValueError,
        match=r"\[ThreadSafeVocabulary\] Seeing a new string 'value2' after calling declareSeenAllStrings().",
    ):
        classifier.evaluate(TEMP_TABULAR_TRAIN_FILE)

    remove_temp_file()


def test_invalid_numeric_column():
    create_temp_file(["colname1,colname2\n", "value1,label1\n", "value3,label2\n"])

    column_datatypes = ["numeric", "label"]
    classifier = bolt.TabularClassifier(
        internal_model_dim=1, n_classes=1, column_datatypes=column_datatypes
    )

    with pytest.raises(
        ValueError,
        match=r"Could not process column 0 as type 'numeric.' Received value: 'value1.'",
    ):
        classifier.train(TEMP_TABULAR_TRAIN_FILE, epochs=1, learning_rate=0.1)

    remove_temp_file()


def test_empty_columns():
    create_temp_file(
        [
            "colname1,colname2,colname3,colname4\n",
            "1,value2,value3,label1\n",
            ",value2,,label2\n",
        ]
    )

    column_datatypes = ["numeric", "categorical", "categorical", "label"]
    classifier = bolt.TabularClassifier(
        internal_model_dim=1, n_classes=2, column_datatypes=column_datatypes
    )

    classifier.train(TEMP_TABULAR_TRAIN_FILE, epochs=1, learning_rate=0.1)
    classifier.evaluate(TEMP_TABULAR_TRAIN_FILE)

    remove_temp_file()


def test_failure_on_new_label_in_testset():
    create_temp_file(["colname1,colname2\n", "value1,label1\n", "value2,label2\n"])

    column_datatypes = ["categorical", "label"]
    classifier = bolt.TabularClassifier(
        internal_model_dim=1, n_classes=2, column_datatypes=column_datatypes
    )

    classifier.train(TEMP_TABULAR_TRAIN_FILE, epochs=1, learning_rate=0.1)

    create_temp_file(["colname1,colname2\n", "value1,label2\n", "value2,label3\n"])

    with pytest.raises(
        ValueError,
        match=r"\[ThreadSafeVocabulary\] Seeing a new string 'label3' after calling.*",
    ):
        classifier.evaluate(TEMP_TABULAR_TRAIN_FILE)

    remove_temp_file()


def test_failure_on_too_many_labels():
    create_temp_file(["colname1,colname2\n", "value1,label1\n", "value2,label2\n"])

    column_datatypes = ["categorical", "label"]
    classifier = bolt.TabularClassifier(
        internal_model_dim=1, n_classes=1, column_datatypes=column_datatypes
    )

    with pytest.raises(
        ValueError, match=r"Expected 1 classes but found an additional class: 'label2."
    ):
        classifier.train(TEMP_TABULAR_TRAIN_FILE, epochs=1, learning_rate=0.1)

    remove_temp_file()


def test_no_label_column():
    create_temp_file(["colname1,colname2\n", "1,value1\n", "2,value2\n"])

    column_datatypes = ["numeric", "categorical"]
    classifier = bolt.TabularClassifier(
        internal_model_dim=1, n_classes=2, column_datatypes=column_datatypes
    )

    with pytest.raises(ValueError, match=r"Dataset does not contain a 'label' column."):
        classifier.train(TEMP_TABULAR_TRAIN_FILE, epochs=1, learning_rate=0.1)

    remove_temp_file()


def test_duplicate_label_column():
    create_temp_file(["colname1,colname2\n", "label1,label1\n", "label2,label2\n"])

    column_datatypes = ["label", "label"]
    classifier = bolt.TabularClassifier(
        internal_model_dim=1, n_classes=2, column_datatypes=column_datatypes
    )

    with pytest.raises(ValueError, match=r"Found multiple 'label' columns in dataset."):
        classifier.train(TEMP_TABULAR_TRAIN_FILE, epochs=1, learning_rate=0.1)

    remove_temp_file()
