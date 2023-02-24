import os
import random
from collections import defaultdict

import pandas as pd
import pytest
from download_dataset_fixtures import download_clinc_dataset
from model_test_utils import compute_evaluate_accuracy
from thirdai import bolt

pytestmark = [pytest.mark.integration]


def test_udt_cold_start_kaggle():
    os.system(
        "curl -L https://www.dropbox.com/s/tf7e5m0cikhcb95/amazon-kaggle-product-catalog-sampled-0.05.csv?dl=0 -o amazon-kaggle-product-catalog.csv"
    )

    catalog_file = "amazon-kaggle-product-catalog.csv"

    df = pd.read_csv(catalog_file)

    model = bolt.UniversalDeepTransformer(
        data_types={
            "QUERY": bolt.types.text(),
            "PRODUCT_ID": bolt.types.categorical(),
        },
        target="PRODUCT_ID",
        n_target_classes=df.shape[0],
        integer_target=True,
    )

    class FinalMetricCallback(bolt.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.ending_train_metric = 0

        def on_train_end(self, model, train_state):
            self.ending_train_metric = train_state.get_train_metric_values(
                "categorical_accuracy"
            )[-1]

    final_metric = FinalMetricCallback()

    model.cold_start(
        filename=catalog_file,
        strong_column_names=["TITLE"],
        weak_column_names=["DESCRIPTION", "BULLET_POINTS", "BRAND"],
        learning_rate=0.001,
        epochs=5,
        metrics=["categorical_accuracy"],
        callbacks=[final_metric],
    )

    os.remove(catalog_file)

    assert final_metric.ending_train_metric > 0.5


def setup_testing_file(missing_values, bad_csv_line):
    filename = "DUMMY_COLDSTART.csv"
    with open(filename, "w") as outfile:
        outfile.write("category,strong,weak1,weak2\n")
        outfile.write("0,this is a title,this is a description,another one\n")

        if missing_values:
            outfile.write("1,there will be no descriptions,,")

        if bad_csv_line:
            outfile.write("1,theres a new line,\n,")

    return filename


def run_coldstart(
    strong_columns=["strong"],
    weak_columns=["weak1", "weak2"],
    validation=None,
    callbacks=[],
    missing_values=False,
    bad_csv_line=False,
    epochs=5,
):
    filename = setup_testing_file(missing_values, bad_csv_line)

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
        match=r"Unable to find column with name 'SOME RANDOM NAME'.",
    ):
        run_coldstart(strong_columns=["SOME RANDOM NAME"])

    with pytest.raises(
        ValueError,
        match=r"Unable to find column with name 'SOME RANDOM NAME'.",
    ):
        run_coldstart(weak_columns=["SOME RANDOM NAME"])


def test_coldstart_empty_strong_or_weak():
    run_coldstart(strong_columns=[])
    run_coldstart(weak_columns=[])


def test_coldstart_missing_values():
    run_coldstart(missing_values=True)


def test_coldstart_bad_csv_line():
    with pytest.raises(
        ValueError,
        match=r"Received a row with a different number of entries than in the header. Expected 4 entries but received 3 entries. Line: 1,theres a new line,",
    ):
        run_coldstart(bad_csv_line=True)
