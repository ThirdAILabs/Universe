import os
import random

import pandas as pd
import pytest
from thirdai import bolt

QUERY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "texts.csv")

pytestmark = [pytest.mark.unit, pytest.mark.release]


def train_model():
    model = bolt.UniversalDeepTransformer(
        data_types={"text": bolt.types.text(), "id": bolt.types.categorical()},
        target="id",
        integer_target=True,
        n_target_classes=100,
        options={
            "extreme_classification": True,
            "extreme_output_dim": 10000,
            "rlhf": True,
        },
    )

    model.train(QUERY_FILE, metrics=["precision@1"])

    model.evaluate(QUERY_FILE, metrics=["precision@1"], use_sparse_inference=True)

    return model


def get_association_samples():
    df = pd.read_csv(QUERY_FILE)

    original_samples = []
    acronym_samples = []
    associations = []

    random_words = []
    for sample in df["text"]:
        words = sample.split()
        random_words.extend(random.choices(words, k=4))

    for sample in df["text"]:
        original_samples.append({"text": sample})

        words = sample.split()
        n_words = len(words)

        start = n_words // 10
        end = n_words // 10 * 9

        acronym = "".join(word[0] for word in words[start:end])
        new_words = (
            words[:start] + [acronym] + words[end:] + random.choices(random_words, k=15)
        )

        new_sample = " ".join(new_words)
        acronym_samples.append({"text": new_sample})

        association = (
            {"text": acronym},
            {"text": " ".join(words[start:end])},
        )
        associations.append(association)

    return original_samples, acronym_samples, associations


def compare_predictions(model, original_samples, acronym_samples):
    correct = 0
    for original, acronym in zip(original_samples, acronym_samples):
        original_pred = model.predict(original)[0][0]
        acronym_pred = model.predict(acronym)[0][0]

        if original_pred == acronym_pred:
            correct += 1

    return correct / len(original_samples)


def test_associate_acronyms():
    model = train_model()

    original_samples, acronym_samples, associations = get_association_samples()

    matches_before_associate = compare_predictions(
        model, original_samples, acronym_samples
    )
    print(matches_before_associate)
    assert matches_before_associate <= 0.5

    model.associate(associations, 4, epochs=10, learning_rate=0.01)

    matches_after_associate = compare_predictions(
        model, original_samples, acronym_samples
    )
    print(matches_after_associate)
    assert matches_after_associate >= 0.9
