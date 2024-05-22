import os
import random
import re

import numpy as np
import pandas as pd
import pytest
from datasets import load_dataset
from thirdai import bolt


def recall(predictions, labels):
    correct = 0
    for preds, label in zip(predictions, labels):
        # There is exactly one label for each sample since there is one correct phrase.
        if label in preds:
            correct += 1
    return correct / len(labels)


def shuffle_chars(word: str) -> str:
    chars = list(word)
    random.shuffle(chars)
    return "".join(chars)


def remove_random_char(word: str) -> str:
    chars = list(word)
    chars.pop(random.randint(0, len(chars) - 1))
    return "".join(chars)


def perturb_sentence(sentence: str) -> str:
    words = sentence.split()

    words_to_modify = np.arange(len(words))
    np.random.shuffle(words_to_modify)
    words_to_modify = words_to_modify[: len(words) // 2]  # Modify half the words

    transformations = random.choices(
        [shuffle_chars, remove_random_char], k=len(words_to_modify)
    )

    for idx, func in zip(words_to_modify, transformations):
        words[idx] = func(words[idx])

    return " ".join(words)


@pytest.fixture(scope="session")
def query_reformulation_dataset():
    df = load_dataset("snips_built_in_intents", "default")["train"].to_pandas()
    df = df.drop("label", axis=1)
    df = df.rename(columns={"text": "correct_query"})

    train_df = df.copy()
    train_df["incorrect_query"] = train_df["correct_query"].apply(perturb_sentence)

    test_df = df.copy()
    test_df["incorrect_query"] = test_df["correct_query"].apply(perturb_sentence)

    inference_samples = []
    inference_labels = []
    for _, row in test_df.iterrows():
        inference_samples.append({"phrase": row["incorrect_query"]})
        inference_labels.append(row["correct_query"])

    return train_df, test_df, (inference_samples, inference_labels)


@pytest.fixture
def train_test_data(request, query_reformulation_dataset):
    train_columns, test_columns = request.param
    train_df, test_df, inference_samples = query_reformulation_dataset

    train_filename = "query_reformulation_train.csv"
    test_filename = "query_reformulation_test.csv"

    train_df.to_csv(train_filename, columns=train_columns, index=False)
    test_df.to_csv(test_filename, columns=test_columns, index=False)

    yield train_filename, test_filename, inference_samples

    os.remove(train_filename)
    os.remove(test_filename)


ALL_COLUMNS = ["incorrect_query", "correct_query"]
CORRECT_ONLY = ["correct_query"]
INCORRECT_ONLY = ["incorrect_query"]


@pytest.mark.unit
@pytest.mark.parametrize(
    "train_test_data, supervised, use_spell_checker",
    [
        ((ALL_COLUMNS, ALL_COLUMNS), True, False),
        ((ALL_COLUMNS, ALL_COLUMNS), True, True),
        ((ALL_COLUMNS, INCORRECT_ONLY), True, False),
        ((ALL_COLUMNS, CORRECT_ONLY), True, False),
        ((CORRECT_ONLY, ALL_COLUMNS), True, False),
        ((CORRECT_ONLY, INCORRECT_ONLY), True, False),
        ((CORRECT_ONLY, CORRECT_ONLY), True, False),
        ((CORRECT_ONLY, CORRECT_ONLY), False, False),
    ],
    indirect=["train_test_data"],
)
def test_query_reformulation(train_test_data, supervised, use_spell_checker):
    train_file, test_file, (inference_samples, inference_labels) = train_test_data

    if supervised:
        model = bolt.UniversalDeepTransformer(
            data_types={
                "incorrect_query": bolt.types.text(),
                "correct_query": bolt.types.text(),
            },
            target="correct_query",
            dataset_size="small",
            use_spell_checker=use_spell_checker,
        )
    else:
        model = bolt.UniversalDeepTransformer(
            data_types={
                "correct_query": bolt.types.text(),
            },
            target="correct_query",
            dataset_size="small",
        )

    model.train(train_file)

    if train_test_data[1] == ALL_COLUMNS:
        metrics = model.evaluate(test_file, top_k=5)
        assert metrics["val_recall"][-1] >= 0.9

    predictions, scores = model.predict_batch(inference_samples, top_k=5)

    assert recall(predictions=predictions, labels=inference_labels) >= 0.9

    for score in scores:
        assert all(a >= b for a, b in zip(score, score[1:]))


@pytest.mark.unit
@pytest.mark.parametrize(
    "query_reformulation_dataset, use_spell_checker",
    [((), True), ((), False)],
    indirect=["query_reformulation_dataset"],
)
def test_query_reformulation_save_load(query_reformulation_dataset, use_spell_checker):
    filename = "./query_reformluation.csv"
    query_reformulation_dataset[0].to_csv(filename, columns=ALL_COLUMNS, index=False)

    model = bolt.UniversalDeepTransformer(
        data_types={
            "incorrect_query": bolt.types.text(),
            "correct_query": bolt.types.text(),
        },
        target="correct_query",
        dataset_size="small",
        use_spell_checker=use_spell_checker,
        n_grams=[2, 3, 4],
    )

    model.train(filename)

    old_metrics = model.evaluate(filename, top_k=1)
    assert old_metrics["val_recall"][-1] >= 0.9

    model_path = "./query_reformulation_model"
    model.save(model_path)
    model = bolt.UniversalDeepTransformer.load(model_path)

    new_metrics = model.evaluate(filename, top_k=1)
    assert new_metrics["val_recall"][-1] >= 0.9

    assert new_metrics["val_recall"][-1] == old_metrics["val_recall"][-1]

    model.train(filename)
    newer_metrics = model.evaluate(filename, top_k=1)
    assert newer_metrics["val_recall"][-1] >= 0.9

    os.remove(filename)
    os.remove(model_path)


@pytest.mark.unit
@pytest.mark.parametrize(
    "query_reformulation_dataset, use_spell_checker",
    [((), True), ((), False)],
    indirect=["query_reformulation_dataset"],
)
def test_query_reformulation_n_grams(query_reformulation_dataset, use_spell_checker):
    filename = "./query_reformluation.csv"
    query_reformulation_dataset[0].to_csv(filename, columns=ALL_COLUMNS, index=False)

    model = bolt.UniversalDeepTransformer(
        data_types={
            "incorrect_query": bolt.types.text(),
            "correct_query": bolt.types.text(),
        },
        target="correct_query",
        dataset_size="small",
        use_spell_checker=use_spell_checker,
        n_grams=[2, 3],
    )

    model.train(filename)

    old_metrics = model.evaluate(filename, top_k=1)
    assert old_metrics["val_recall"][-1] >= 0.85

    os.remove(filename)


@pytest.mark.unit
@pytest.mark.parametrize("use_spell_checker", [(True), (False)])
def test_query_reformulation_throws_error_wrong_argument(use_spell_checker):
    with pytest.raises(
        ValueError,
        match=re.escape(f"n_grams argument must contain only positive integers"),
    ):
        model = bolt.UniversalDeepTransformer(
            data_types={
                "incorrect_query": bolt.types.text(),
                "correct_query": bolt.types.text(),
            },
            target="correct_query",
            dataset_size="small",
            use_spell_checker=use_spell_checker,
            n_grams=[-1, 3],
        )

    with pytest.raises(
        ValueError,
        match=re.escape(f"Expected parameter 'n_grams' to have type List[int]."),
    ):
        model = bolt.UniversalDeepTransformer(
            data_types={
                "incorrect_query": bolt.types.text(),
                "correct_query": bolt.types.text(),
            },
            target="correct_query",
            dataset_size="small",
            use_spell_checker=use_spell_checker,
            n_grams=1,
        )
