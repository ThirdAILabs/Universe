import platform

import numpy as np
import pandas as pd
import pytest
from download_dataset_fixtures import download_census_income
from thirdai import bolt
from thirdai.demos import to_udt_input_batch

pytestmark = [pytest.mark.unit, pytest.mark.release]


METADATA_FILE = "metadata.csv"
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
KEY_COLUMN_NAME = "id"
USER_COLUMN_NAME = "user"
ITEM_COLUMN_NAME = "item"
LABEL_COLUMN_NAME = "label"
TS_COLUMN_NAME = "timestamp"
DELIMITER = "}"  # Make sure custom delimiters work


def write_metadata_file(orig_train_df, orig_test_df):
    all_df = pd.concat([orig_train_df, orig_test_df], ignore_index=True)
    metadata_columns = all_df.columns[:-1]  # Exclude label column
    metadata_df = all_df[metadata_columns]
    metadata_df[KEY_COLUMN_NAME] = pd.Series(np.arange(len(all_df)))
    metadata_df.head(len(orig_train_df)).to_csv(
        METADATA_FILE, index=False, sep=DELIMITER
    )
    return metadata_df.tail(len(orig_test_df))


def curate_from_census_income_dataset(orig_train_df, orig_test_df, curate_metadata_for):
    test_metadata = write_metadata_file(orig_train_df, orig_test_df)

    train_id_series = pd.Series(np.arange(len(orig_train_df)))
    test_id_series = pd.Series(
        np.arange(len(orig_train_df), len(orig_train_df) + len(orig_test_df)),
    )

    if curate_metadata_for == "user":
        train_df = pd.DataFrame.from_dict(
            {
                USER_COLUMN_NAME: train_id_series,
                LABEL_COLUMN_NAME: orig_train_df[LABEL_COLUMN_NAME],
            }
        )
        test_df = pd.DataFrame.from_dict(
            {
                USER_COLUMN_NAME: test_id_series,
                LABEL_COLUMN_NAME: orig_test_df[LABEL_COLUMN_NAME],
            }
        )
    elif curate_metadata_for == "item":
        train_df = pd.DataFrame.from_dict(
            {
                USER_COLUMN_NAME: pd.Series(
                    np.zeros(len(orig_train_df), dtype=np.int32)
                ),
                ITEM_COLUMN_NAME: train_id_series,
                LABEL_COLUMN_NAME: orig_train_df[LABEL_COLUMN_NAME],
                TS_COLUMN_NAME: pd.Series(
                    ["2022-02-02" for _ in range(len(orig_train_df))]
                ),
            }
        )
        test_df = pd.DataFrame.from_dict(
            {
                USER_COLUMN_NAME: pd.Series(
                    np.zeros(len(orig_test_df), dtype=np.int32)
                ),
                ITEM_COLUMN_NAME: test_id_series,
                LABEL_COLUMN_NAME: orig_test_df[LABEL_COLUMN_NAME],
                TS_COLUMN_NAME: pd.Series(
                    ["2022-02-02" for _ in range(len(orig_test_df))]
                ),
            }
        )
    else:
        raise ValueError(
            curate_metadata_for + " is not a valid option to curate metadata for"
        )
    train_df.to_csv(TRAIN_FILE, index=False)
    test_df.to_csv(TEST_FILE, index=False)

    return test_metadata


def make_trained_model_with_metadata(n_samples, metadata_src):
    metadata = bolt.types.metadata(
        filename=METADATA_FILE,
        key_column_name=KEY_COLUMN_NAME,
        data_types={
            "age": bolt.types.numerical(range=(17, 90)),
            "workclass": bolt.types.categorical(),
            "fnlwgt": bolt.types.numerical(range=(12285, 1484705)),
            "education": bolt.types.categorical(),
            "education-num": bolt.types.categorical(),
            "marital-status": bolt.types.categorical(),
            "occupation": bolt.types.categorical(),
            "relationship": bolt.types.categorical(),
            "race": bolt.types.categorical(),
            "sex": bolt.types.categorical(),
            "capital-gain": bolt.types.numerical(range=(0, 99999)),
            "capital-loss": bolt.types.numerical(range=(0, 4356)),
            "hours-per-week": bolt.types.numerical(range=(1, 99)),
            "native-country": bolt.types.categorical(),
        },
        delimiter=DELIMITER,
    )

    if metadata_src == "user":
        data_types = {
            USER_COLUMN_NAME: bolt.types.categorical(metadata=metadata),
            LABEL_COLUMN_NAME: bolt.types.categorical(),
        }
        temporal = {}
    elif metadata_src == "item":
        data_types = {
            USER_COLUMN_NAME: bolt.types.categorical(),
            ITEM_COLUMN_NAME: bolt.types.categorical(metadata=metadata),
            LABEL_COLUMN_NAME: bolt.types.categorical(),
            TS_COLUMN_NAME: bolt.types.date(),
        }
        temporal = {
            USER_COLUMN_NAME: [
                bolt.temporal.categorical(
                    ITEM_COLUMN_NAME,
                    track_last_n=1,
                    column_known_during_inference=True,
                    use_metadata=True,
                )
            ]
        }
    else:
        raise ValueError(metadata_src + " is not a valid metadata_src")

    model = bolt.UniversalDeepTransformer(
        data_types,
        temporal_tracking_relationships=temporal,
        target="label",
        n_target_classes=2,
    )

    model.train(TRAIN_FILE, epochs=3, learning_rate=0.01, verbose=False)

    return model


def get_ground_truths(trained_model, original_test_df):
    n_classes = len(original_test_df["label"].unique())
    classes = {}
    for i in range(n_classes):
        classes[trained_model.class_name(i)] = i
    ground_truth = original_test_df["label"].map(classes).to_numpy()
    return ground_truth


def get_accuracy_on_test_data(trained_model, original_test_df):
    metrics = trained_model.evaluate(TEST_FILE, metrics=["categorical_accuracy"])

    return metrics["val_categorical_accuracy"][-1]


def index_test_metadata(
    model, index_metadata_option, metadata_column_name, test_metadata
):
    if index_metadata_option is None:
        return

    update_batch = to_udt_input_batch(test_metadata)
    if index_metadata_option == "single":
        for update in update_batch:
            model.index_metadata(metadata_column_name, update)
    elif index_metadata_option == "batch":
        model.index_metadata_batch(metadata_column_name, update_batch)


def run_metadata_test(metadata_src, index_metadata_option, download_census_income):
    """Metadata support allows us to preprocess vectors from a metadata file
    that corresponds with a categorical column in the main dataset. When we load
    the main dataset, by appending the corresponding preprocessed vector.

    For example, we may have a metadata file with the following columns:
    user,feature_1,feature_2

    And a main dataset with the following columns:
    timestamp,movie,user

    For each row in the main dataset, we will append vectors from the metadata
    file that corresponds to the user in the row.

    To test this, we take a tabular dataset and move every column except the
    label column to a metadata file. We also add an "id" column whose values are
    row indices. The main dataset will only consist of "id" and label columns.
    Thus, the model can only learn properly if it successfully uses metadata.
    """
    orig_train_file, orig_test_file, _ = download_census_income

    train_df = pd.read_csv(orig_train_file)
    test_df = pd.read_csv(orig_test_file)

    test_metadata = curate_from_census_income_dataset(
        train_df,
        test_df,
        curate_metadata_for=metadata_src,
    )

    model = make_trained_model_with_metadata(
        n_samples=len(pd.concat([train_df, test_df])), metadata_src=metadata_src
    )

    # Accuracy should be low without indexing test metadata
    acc = get_accuracy_on_test_data(model, test_df)
    assert acc < 0.8

    # Index test metadata
    metadata_column_name = (
        USER_COLUMN_NAME if metadata_src == "user" else ITEM_COLUMN_NAME
    )
    update_batch = to_udt_input_batch(test_metadata)
    if index_metadata_option == "single":
        for update in update_batch:
            model.index_metadata(metadata_column_name, update)
    elif index_metadata_option == "batch":
        model.index_metadata_batch(metadata_column_name, update_batch)

    # Accuracy should increase after indexing test metadata
    acc = get_accuracy_on_test_data(model, test_df)
    assert acc > 0.83


def test_item_metadata_single_indexing(download_census_income):
    run_metadata_test(
        metadata_src="item",
        index_metadata_option="single",
        download_census_income=download_census_income,
    )


def test_user_metadata_batch_indexing(download_census_income):
    run_metadata_test(
        metadata_src="user",
        index_metadata_option="batch",
        download_census_income=download_census_income,
    )


def test_item_metadata_batch_indexing(download_census_income):
    run_metadata_test(
        metadata_src="item",
        index_metadata_option="batch",
        download_census_income=download_census_income,
    )


def test_user_metadata_single_indexing(download_census_income):
    run_metadata_test(
        metadata_src="user",
        index_metadata_option="single",
        download_census_income=download_census_income,
    )
