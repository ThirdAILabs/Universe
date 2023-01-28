import os
import random
from collections import defaultdict

import pandas as pd
import pytest
from download_dataset_fixtures import download_clinc_dataset
from model_test_utils import compute_evaluate_accuracy
from thirdai import bolt


@pytest.fixture(scope="module")
def cold_start_dataset(download_clinc_dataset):
    """
    This constructs a dataset for cold start where the strong column is the original
    query for each sample and the weak column is another query that maps to the
    same label.
    """
    COLD_START_TRAIN_FILE = "./clinc_cold_start.csv"
    TEXT_COLUMN_NAME = "text"
    WEAK_COLUMN_NAME = "additional_text"

    train_filename, _, _ = download_clinc_dataset

    df = pd.read_csv(train_filename)

    classes = defaultdict(list)
    for _, row in df.iterrows():
        classes[row["category"]].append(row[TEXT_COLUMN_NAME])

    additional_text = []

    for _, row in df.iterrows():
        additional_text.append(random.choice(classes[row["category"]]))

    df[WEAK_COLUMN_NAME] = additional_text

    df.to_csv(COLD_START_TRAIN_FILE, index=False)

    return COLD_START_TRAIN_FILE, TEXT_COLUMN_NAME, WEAK_COLUMN_NAME


def test_udt_cold_start(download_clinc_dataset, cold_start_dataset):
    _, test_filename, inference_samples = download_clinc_dataset
    cold_start_filename, text_column_name, weak_column_name = cold_start_dataset

    model = bolt.UniversalDeepTransformer(
        data_types={
            "category": bolt.types.categorical(),
            "text": bolt.types.text(),
        },
        target="category",
        n_target_classes=150,
        integer_target=True,
    )

    model.cold_start(
        filename=cold_start_filename,
        strong_column_names=[text_column_name],
        weak_column_names=[weak_column_name],
        learning_rate=0.01,
    )

    # We need to train on a file so that UDT initializes the dataset loader.
    # Initializing the dataset loaders requires knowing the order of the columns
    # in the CSV file which UDT parses during the first call to train.
    empty_train_file = "./empty_clinc.csv"
    with open(empty_train_file, "w") as file:
        file.write("category,text\n")
        file.write(
            "131,what expression would i use to say i love you if i were an italian\n"
        )

    model.train(empty_train_file, epochs=1, learning_rate=0.01)

    os.remove(empty_train_file)

    acc = compute_evaluate_accuracy(
        model, test_filename, inference_samples, use_class_name=False
    )

    # Accuracy is around 78-80%, with regular training it is a few percent lower.
    assert acc >= 0.7


def setup_testing_file(missing_values):
    filename = "DUMMY_COLDSTART.csv"
    with open(filename, "w") as outfile:
        outfile.write("category,strong,weak1,weak2\n")
        outfile.write("0,this is a title,this is a description,another one\n")

        if missing_values:
            outfile.write("1,there will be no descriptions,,")

    return filename


def run_coldstart(
    strong_columns=["strong"],
    weak_columns=["weak1", "weak2"],
    validation=None,
    callbacks=[],
    missing_values=False,
    epochs=5,
):
    filename = setup_testing_file(missing_values)

    model = bolt.UniversalDeepTransformer(
        data_types={
            "category": bolt.types.categorical(),
            "text": bolt.types.text(),
        },
        target="category",
        n_target_classes=2,
        integer_target=True,
    )

    model.cold_start(
        filename=filename,
        strong_column_names=strong_columns,
        weak_column_names=weak_columns,
        learning_rate=0.01,
        epochs=epochs,
        validation=validation,
        callbacks=callbacks,
    )

    os.remove(filename)


def test_coldstart_validation():
    val_filename = "val_file.csv"
    with open(val_filename, "x") as val_file:
        val_file.write("category,text\n")
        val_file.write("1,some text here\n")

    validation = bolt.Validation(
        filename=val_filename, interval=4, metrics=["categorical_accuracy"]
    )

    run_coldstart(validation=validation)

    os.remove(val_filename)


def test_coldstart_callbacks():
    class CountCallback(bolt.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.epoch_count = 0

        def on_epoch_end(self, model, train_state):
            self.epoch_count += 1

    count_callback = CountCallback()

    run_coldstart(callbacks=[count_callback], epochs=5)

    assert count_callback.epoch_count == 5


def test_coldstart_missing_strong_or_weak():
    with pytest.raises(
        ValueError,
        match=r"Column SOME RANDOM NAME not found in dataset.",
    ):
        run_coldstart(strong_columns=["SOME RANDOM NAME"])


    with pytest.raises(
        ValueError,
        match=r"Column SOME RANDOM NAME not found in dataset.",
    ):
        run_coldstart(weak_columns=["SOME RANDOM NAME"])


def test_coldstart_empty_strong_or_weak():
    run_coldstart(strong_columns=[])
    run_coldstart(weak_columns=[])


def test_coldstart_missing_values():
    run_coldstart(missing_values=True)
