import os
import random

import pandas as pd
import pytest
from mach_retriever_utils import QUERY_FILE, train_simple_mach_retriever
from thirdai import bolt, data

pytestmark = [pytest.mark.unit, pytest.mark.release]


def get_upvote_samples():
    df = pd.read_csv(QUERY_FILE)

    acronyms = df["text"].map(lambda s: "".join(w[0] for w in s.split())).to_list()
    ids = df["id"].to_list()

    acronyms = data.ColumnMap({"text": data.columns.StringColumn(acronyms)})
    upvotes = data.ColumnMap(
        {
            "text": acronyms["text"],
            "id": data.columns.TokenColumn(ids, dim=0xFFFFFFFF),
        }
    )

    return ids, acronyms, upvotes


def get_association_samples():
    df = pd.read_csv(QUERY_FILE)

    acronyms = []
    sources = []
    targets = []

    random_words = []
    for sample in df["text"]:
        words = sample.split()
        random_words.extend(random.choices(words, k=4))

    for sample in df["text"]:
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

        acronyms.append(" ".join(new_words))

        sources.append(acronym)
        targets.append(" ".join(words[start:end]))

    ids = df["id"].to_list()

    acronyms = data.ColumnMap({"text": data.columns.StringColumn(acronyms)})
    sources = data.ColumnMap({"text": data.columns.StringColumn(sources)})
    targets = data.ColumnMap({"text": data.columns.StringColumn(targets)})

    return ids, acronyms, sources, targets


def accuracy(model, correct_labels, samples):
    predictions = model.search(samples, top_k=1)
    correct = 0
    for preds, label in zip(predictions, correct_labels):
        if preds[0][0] == label:
            correct += 1

    return correct / len(correct_labels)


@pytest.mark.parametrize("serialize", [True, False])
def test_mach_retriever_upvote(serialize):
    model = train_simple_mach_retriever()

    if serialize:
        # This is just a simple addition to ensure that balancing samples and related
        # state are saved correctly, and that a loaded model can be associated/upvoted.
        path = "./test-mach-retriever-rlhf-model"
        model.save(path)
        model = bolt.MachRetriever.load(path)
        os.remove(path)

    correct_labels, acronym_samples, upvotes = get_upvote_samples()

    acc_before_upvote = accuracy(model, correct_labels, acronym_samples)
    print(acc_before_upvote)
    assert acc_before_upvote <= 0.3  # Should be close to 0

    model.upvote(upvotes, learning_rate=0.01, epochs=2)

    acc_after_upvote = accuracy(model, correct_labels, acronym_samples)
    print(acc_after_upvote)
    assert acc_after_upvote >= 0.9  # Should be close to 0.99


@pytest.mark.parametrize("serialize", [True, False])
def test_mach_retriever_associate(serialize):
    model = train_simple_mach_retriever()

    if serialize:
        # This is just a simple addition to ensure that balancing samples and related
        # state are saved correctly, and that a loaded model can be associated/upvoted.
        path = "./test-mach-retriever-rlhf-model"
        model.save(path)
        model = bolt.MachRetriever.load(path)
        os.remove(path)

    correct_labels, acronym_samples, sources, targets = get_association_samples()

    acc_before_associate = accuracy(model, correct_labels, acronym_samples)
    print(acc_before_associate)
    assert acc_before_associate <= 0.5

    model.associate(
        sources=sources, targets=targets, n_buckets=4, epochs=10, learning_rate=0.01
    )

    acc_after_associate = accuracy(model, correct_labels, acronym_samples)
    print(acc_after_associate)
    assert acc_after_associate >= 0.9
