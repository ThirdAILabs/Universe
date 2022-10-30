import numpy as np
import pandas as pd
from thirdai import bolt, deployment
import os

CENSUS_INCOME_BASE_DOWNLOAD_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/"
)

ORIGINAL_TRAIN_FILE = "./census_income_train.csv"
ORIGINAL_TEST_FILE = "./census_income_test.csv"

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
    if not os.path.exists(ORIGINAL_TRAIN_FILE):
        os.system(
            f"curl {CENSUS_INCOME_BASE_DOWNLOAD_URL}adult.data --output {TRAIN_FILE}"
        )

    if not os.path.exists(ORIGINAL_TEST_FILE):
        os.system(
            f"curl {CENSUS_INCOME_BASE_DOWNLOAD_URL}adult.test --output {TEST_FILE}"
        )
        # reformat the test file
        with open(ORIGINAL_TEST_FILE, "r") as file:
            data = file.read().splitlines(True)
        with open(ORIGINAL_TEST_FILE, "w") as file:
            # for some reason each of the labels end with a "." in the test set
            # loop through data[1:] since the first line is bogus
            file.writelines([line.replace(".", "") for line in data[1:]])


METADATA_FILENAME = "metadata.csv"
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
USER_COLUMN_NAME = "user"
ITEM_COLUMN_NAME = "item"
LABEL_COLUMN_NAME = "label"
KEY_COLUMN_NAME = "id"
TS_COLUMN_NAME = "timestamp"


def write_metadata_file(orig_train_df, orig_test_df):
    all_df = pd.concat([orig_train_df, orig_test_df], ignore_index=True)
    metadata_columns = COLUMN_NAMES[:-1]
    metadata_df = all_df[metadata_columns]
    metadata_df[KEY_COLUMN_NAME] = pd.Series(np.arange(len(all_df)))
    metadata_df.to_csv(METADATA_FILENAME, index=False)


def curate_from_census_income_dataset(curate_metadata_for):
    orig_train_df = pd.read_csv(ORIGINAL_TRAIN_FILE, header=None)
    orig_train_df.columns = COLUMN_NAMES
    orig_test_df = pd.read_csv(ORIGINAL_TEST_FILE, header=None)
    orig_test_df.columns = COLUMN_NAMES

    write_metadata_file(orig_train_df, orig_test_df)

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


def make_metadata():
    return bolt.types.metadata(
        filename=METADATA_FILENAME,
        key_column_name="id",
        data_types={
            # "age": bolt.types.numerical(range=(17, 90)),
            "workclass": bolt.types.categorical(n_unique_classes=9),
            # "fnlwgt": bolt.types.numerical(range=(12285, 1484705)),
            "education": bolt.types.categorical(n_unique_classes=16),
            "education-num": bolt.types.categorical(n_unique_classes=16),
            "marital-status": bolt.types.categorical(n_unique_classes=7),
            "occupation": bolt.types.categorical(n_unique_classes=15),
            "relationship": bolt.types.categorical(n_unique_classes=6),
            "race": bolt.types.categorical(n_unique_classes=5),
            "sex": bolt.types.categorical(n_unique_classes=2),
            # "capital-gain": bolt.types.numerical(range=(0, 99999)),
            # "capital-loss": bolt.types.numerical(range=(0, 4356)),
            # "hours-per-week": bolt.types.numerical(range=(1, 99)),
            "native-country": bolt.types.categorical(n_unique_classes=42),
        },
    )


def make_trained_model_with_metadata(metadata_src):
    n_unique_ids = len(pd.concat([pd.read_csv(TRAIN_FILE), pd.read_csv(TEST_FILE)]))
    if metadata_src == "user":
        data_types = {
            USER_COLUMN_NAME: bolt.types.categorical(
                n_unique_classes=n_unique_ids, metadata=make_metadata()
            ),
            LABEL_COLUMN_NAME: bolt.types.categorical(n_unique_classes=2),
        }
        temporal = {}
    elif metadata_src == "item":
        data_types = {
            USER_COLUMN_NAME: bolt.types.categorical(n_unique_classes=1),
            ITEM_COLUMN_NAME: bolt.types.categorical(
                n_unique_classes=n_unique_ids, metadata=make_metadata()
            ),
            LABEL_COLUMN_NAME: bolt.types.categorical(n_unique_classes=2),
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

    model = deployment.UniversalDeepTransformer(
        data_types, temporal_tracking_relationships=temporal, target="label"
    )

    train_config = bolt.graph.TrainConfig.make(
        epochs=3, learning_rate=0.01
    ).with_metrics(["categorical_accuracy"])

    model.train(TRAIN_FILE, train_config)

    return model


def get_n_classes(dataframe):
    return len(dataframe["label"].unique())


def get_ground_truths(trained_model, test_file):
    df = pd.read_csv(test_file)
    n_classes = get_n_classes(df)
    classes = {}
    for i in range(n_classes):
        classes[trained_model.class_name(i)] = i
    ground_truth = df["label"].map(classes).to_numpy()
    return ground_truth


def get_accuracy_on_test_data(trained_model, test_file):

    results = trained_model.evaluate(test_file)
    result_ids = np.argmax(results, axis=1)
    ground_truth = get_ground_truths(trained_model, test_file)

    return sum(result_ids == ground_truth) / len(result_ids)


def test_metadata():
    for metadata_src in ["user", "item"]:
        curate_from_census_income_dataset(curate_metadata_for=metadata_src)

        model = make_trained_model_with_metadata(metadata_src=metadata_src)

        acc = get_accuracy_on_test_data(model, TEST_FILE)

        assert acc > 0.8

        os.remove(TRAIN_FILE)
        os.remove(TEST_FILE)
        os.remove(METADATA_FILENAME)
