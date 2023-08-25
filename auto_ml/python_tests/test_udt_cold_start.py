import os

import pytest
from download_dataset_fixtures import download_amazon_kaggle_product_catalog_sampled
from thirdai import bolt

pytestmark = [pytest.mark.unit]


def test_udt_cold_start_kaggle(download_amazon_kaggle_product_catalog_sampled):
    catalog_file, n_target_classes = download_amazon_kaggle_product_catalog_sampled

    model = bolt.UniversalDeepTransformer(
        data_types={
            "QUERY": bolt.types.text(),
            "PRODUCT_ID": bolt.types.categorical(),
        },
        target="PRODUCT_ID",
        n_target_classes=n_target_classes,
        integer_target=True,
    )

    metrics = model.cold_start(
        filename=catalog_file,
        strong_column_names=["TITLE"],
        weak_column_names=["DESCRIPTION", "BULLET_POINTS", "BRAND"],
        learning_rate=0.001,
        epochs=5,
        batch_size=2000,
        metrics=["categorical_accuracy"],
        shuffle_reservoir_size=32000,
    )

    os.remove(catalog_file)

    assert metrics["train_categorical_accuracy"][-1] > 0.5


def setup_testing_file(missing_values, bad_csv_line, integer_target=False):
    filename = "DUMMY_COLDSTART.csv"
    with open(filename, "w") as outfile:
        outfile.write("category,strong,weak1,weak2\n")
        outfile.write("0,this is a title,this is a description,another one\n")

        if missing_values:
            outfile.write("1,there will be no descriptions,,")

        if bad_csv_line:
            outfile.write("1,theres a new line,\n,")

        if not integer_target:
            outfile.write("LMFAO,this is not an integer,,\n")

    return filename


def run_coldstart(
    strong_columns=["strong"],
    weak_columns=["weak1", "weak2"],
    validation=None,
    missing_values=False,
    bad_csv_line=False,
    epochs=5,
    integer_target=True,
):
    filename = setup_testing_file(missing_values, bad_csv_line, integer_target)

    model = bolt.UniversalDeepTransformer(
        data_types={
            "category": bolt.types.categorical(),
            "text": bolt.types.text(),
        },
        target="category",
        n_target_classes=3,
        integer_target=integer_target,
    )

    model.cold_start(
        filename=filename,
        strong_column_names=strong_columns,
        weak_column_names=weak_columns,
        learning_rate=0.01,
        epochs=epochs,
        validation=validation,
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
    with pytest.raises(ValueError, match=r"Expected 4 columns. But received row.*"):
        run_coldstart(bad_csv_line=True)


@pytest.mark.parametrize("integer_target", [True, False])
def test_coldstart_target_type(integer_target):
    run_coldstart(integer_target=integer_target)
