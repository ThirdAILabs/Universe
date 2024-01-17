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

    model.train(QUERY_FILE, metrics=["precision@1"], verbose=False)

    model.evaluate(QUERY_FILE, metrics=["precision@1"], use_sparse_inference=True)

    return model


# TODO(Any): add another version of this test that replaces words with synonyms
# to test how assoicating between the synonyms transfers between different samples
# which have occurrences of the synonym.


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

        # Replace the middle words with the acronym and add random words selected
        # from some of the other samples. These random words just make the sample
        # more difficult for the model so that it can't just get the answer by looking
        # at the words that are not part of the acronym.
        new_words = (
            words[:start] + [acronym] + words[end:] + random.choices(random_words, k=15)
        )

        new_sample = " ".join(new_words)
        acronym_samples.append({"text": new_sample})

        association = (acronym, " ".join(words[start:end]))
        associations.append(association)

    return original_samples, acronym_samples, associations


def compare_predictions(model, original_samples, acronym_samples):
    correct = 0
    original_preds = model.predict_batch(original_samples)
    acronym_preds = model.predict_batch(acronym_samples)
    for original_pred, acronym_pred in zip(original_preds, acronym_preds):
        if original_pred[0][0] == acronym_pred[0][0]:
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

    model.associate(associations, n_buckets=4, epochs=10, learning_rate=0.01)

    matches_after_associate = compare_predictions(
        model, original_samples, acronym_samples
    )
    print(matches_after_associate)
    assert matches_after_associate >= 0.9


def test_associate_train_acronyms():
    model = train_model()

    original_samples, acronym_samples, associations = get_association_samples()

    matches_before_associate = compare_predictions(
        model, original_samples, acronym_samples
    )
    print(matches_before_associate)
    assert matches_before_associate <= 0.5

    model.associate_train(
        filename=QUERY_FILE,
        source_target_samples=associations,
        n_buckets=4,
        n_association_samples=4,
        epochs=20,  # We need more epochs in this test because we don't do the same sample replication
        learning_rate=0.01,
        verbose=False,
        batch_size=100,
    )

    matches_after_associate = compare_predictions(
        model, original_samples, acronym_samples
    )

    print(matches_after_associate)  # Accuracy should be around 0.98-1.0
    assert matches_after_associate >= 0.9


def get_upvote_samples():
    df = pd.read_csv(QUERY_FILE)
    df["acronym"] = df["text"].map(lambda s: "".join(w[0] for w in s.split()))

    original_samples = []
    acronyms = []
    upvotes = []

    for _, row in df.iterrows():
        original_samples.append({"text": row["text"]})
        acronyms.append({"text": row["acronym"]})
        upvotes.append((row["acronym"], row["id"]))

    return original_samples, acronyms, upvotes


def test_upvote():
    # This test trains a mach model on a simple dataset of 100 articles from ag-news.
    # Then it creates "acronym" samples which are just the concatenation of the
    # first letter of each word of each article. It checks that originally the
    # accuracy on this new dataset is bad, then upvotes the acronym samples with
    # the correct labels and checks that the accuracy improves.

    model = train_model()

    original_samples, acronym_samples, upvotes = get_upvote_samples()

    matches_before_upvote = compare_predictions(
        model, original_samples, acronym_samples
    )
    print(matches_before_upvote)
    assert matches_before_upvote <= 0.2  # Should be close to 0

    model.upvote(upvotes, learning_rate=0.01, epochs=2)

    matches_after_upvote = compare_predictions(model, original_samples, acronym_samples)
    print(matches_after_upvote)
    assert matches_after_upvote >= 0.9  # Should be close to 0.99
