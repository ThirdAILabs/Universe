from thirdai import bolt
from datasets import load_dataset
import pytest
import numpy as np
import pandas as pd
import random
import os


def recall(predictions, labels):
    correct = 0
    for preds, label in zip(predictions, labels):
        if label in preds:
            correct += 1
    return correct / len(labels)


def shuffle_chars(word: str) -> str:
    chars = list(word)
    random.shuffle(chars)
    return "".join(chars)


def remove_char(word: str) -> str:
    chars = list(word)
    chars.pop(random.randint(0, len(chars) - 1))
    return "".join(chars)


def perturb_sentence(sentence: str) -> str:
    words = sentence.split()

    words_to_modify = np.arange(len(words))
    np.random.shuffle(words_to_modify)
    words_to_modify = words_to_modify[: len(words) // 2]  # Modify half the words

    transformations = random.choices(
        [shuffle_chars, remove_char], k=len(words_to_modify)
    )

    for idx, func in zip(words_to_modify, transformations):
        words[idx] = func(words[idx])

    return " ".join(words)


@pytest.fixture(scope="session")
def query_reformulation_dataset():
    df = load_dataset("snips_built_in_intents", "small")["train"].to_pandas()
    df = df.drop("label", axis=1)
    df = df.rename(columns={"text": "correct_query"})

    train_df = pd.DataFrame.copy(df)
    train_df["incorrect_query"] = train_df["correct_query"].apply(perturb_sentence)

    test_df = pd.DataFrame.copy(df)
    test_df["incorrect_query"] = test_df["correct_query"].apply(perturb_sentence)

    inference_samples = []
    inference_labels = []
    for _, row in test_df.iterrows():
        inference_samples.append({"phrase": row["incorrect_query"]})
        inference_labels.append(row["correct_query"])

    return train_df, test_df, (inference_samples, inference_labels)


@pytest.fixture
def train_test_data(request, query_reformulation_dataset):
    # request.params is a pair in which the first elmement is the columns to have
    # in the train dataset, and the second element is the columns to have in the
    # test dataset.
    train_df, test_df, inference_samples = query_reformulation_dataset

    train_filename = "query_reformulation_train.csv"
    test_filename = "query_reformulation_test.csv"

    train_df.to_csv(train_filename, columns=request.param[0], index=False)
    test_df.to_csv(test_filename, columns=request.param[1], index=False)

    yield train_filename, test_filename, inference_samples

    os.remove(train_filename)
    os.remove(test_filename)


ALL_COLUMNS = ["incorrect_query", "correct_query"]
CORRECT_ONLY = ["correct_query"]
INCORRECT_ONLY = ["incorrect_query"]


@pytest.mark.unit
@pytest.mark.parametrize(
    "train_test_data, supervised",
    [
        ((ALL_COLUMNS, ALL_COLUMNS), True),
        ((ALL_COLUMNS, INCORRECT_ONLY), True),
        ((ALL_COLUMNS, CORRECT_ONLY), True),
        ((CORRECT_ONLY, ALL_COLUMNS), True),
        ((CORRECT_ONLY, INCORRECT_ONLY), True),
        ((CORRECT_ONLY, CORRECT_ONLY), True),
        ((CORRECT_ONLY, CORRECT_ONLY), False),
    ],
    indirect=["train_test_data"],
)
def test_query_reformulation(train_test_data, supervised):
    train_file, test_file, (inference_samples, inference_labels) = train_test_data

    if supervised:
        model = bolt.UniversalDeepTransformer(
            source_column="incorrect_query",
            target_column="correct_query",
            dataset_size="small",
        )
    else:
        model = bolt.UniversalDeepTransformer(
            target_column="correct_query",
            dataset_size="small",
        )

    model.train(train_file)

    eval_predictions, _ = model.evaluate(test_file, top_k=5)

    assert recall(predictions=eval_predictions, labels=inference_labels) >= 0.9

    predictions, scores = model.predict_batch(inference_samples, top_k=5)

    assert recall(predictions=predictions, labels=inference_labels) >= 0.9

    for score in scores:
        assert all(a >= b for a, b in zip(score, score[1:]))


def test_query_reformulation_save_load(query_reformulation_dataset):
    filename = "./query_reformluation.csv"
    query_reformulation_dataset[0].to_csv(filename, columns=ALL_COLUMNS, index=False)

    model = bolt.UniversalDeepTransformer(
        source_column="incorrect_query",
        target_column="correct_query",
        dataset_size="small",
    )

    model.train(filename)

    old_predictions = model.evaluate(filename, top_k=2)

    model_path = "./query_reformulation_model"
    model.save(model_path)
    model = bolt.UniversalDeepTransformer.load(model_path)

    new_predictions = model.evaluate(filename, top_k=2)
    assert old_predictions == new_predictions

    model.train(filename)
    newer_predictions = model.evaluate(filename, top_k=2)
    # The scores may be different because we've inserted duplicate elements
    assert old_predictions[0] == newer_predictions[0]
