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
    COLD_START_TRAIN_FILE = "./clinc_cold_start.csv"
    WEAK_COLUMN_NAME = "additional_text"

    train_filename, _, _ = download_clinc_dataset

    df = pd.read_csv(train_filename)

    classes = defaultdict(list)
    for _, row in df.iterrows():
        classes[row["category"]].append(row["text"])

    additional_text = []

    for _, row in df.iterrows():
        additional_text.append(random.choice(classes[row["category"]]))

    df[WEAK_COLUMN_NAME] = additional_text

    df.to_csv(COLD_START_TRAIN_FILE, index=False)

    return COLD_START_TRAIN_FILE, WEAK_COLUMN_NAME


def test_udt_cold_start(download_clinc_dataset, cold_start_dataset):
    _, test_filename, inference_samples = download_clinc_dataset
    cold_start_filename, weak_column = cold_start_dataset

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
        strong_column_names=["text"],
        weak_column_names=[weak_column],
        learning_rate=0.01,
    )

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
