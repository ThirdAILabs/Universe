import os
import random

import pandas as pd
import pytest
from ndbv2_utils import compute_accuracy, load_chunks
from thirdai.neural_db_v2.core.types import ChunkBatch, SupervisedBatch
from thirdai.neural_db_v2.retrievers import FinetunableRetriever

pytestmark = [pytest.mark.unit]


def build_retriever(chunk_df: pd.DataFrame) -> FinetunableRetriever:
    chunk_batches = []
    batch_size = 25
    for i in range(0, len(chunk_df), batch_size):
        chunks = chunk_df.iloc[i : i + batch_size]
        chunk_batches.append(
            ChunkBatch(
                text=chunks["text"],
                keywords=pd.Series(["" for _ in range(len(chunks))]),
                chunk_id=chunks["id"],
            )
        )

    retriever = FinetunableRetriever()

    retriever.insert(chunk_batches)

    return retriever


def subsample_query(text, k=20):
    return " ".join(random.choices(text.split(), k=k))


def check_basic_query_accuracy(retriever, dataset):
    random.seed(64)
    for row in dataset.itertuples():
        query = subsample_query(row.text)
        assert retriever.search([query], top_k=1)[0][0][0] == row.id


def test_finetunable_retriever_search(load_chunks):
    retriever = build_retriever(load_chunks)

    check_basic_query_accuracy(retriever, load_chunks)

    for _, row in load_chunks.iterrows():
        id = row["id"]
        rank_results = retriever.rank(
            [row["text"]], choices=[set([id + 2, id, id + 1])], top_k=1
        )
        assert id == rank_results[0][0][0]


def test_finetunable_retriever_delete(load_chunks):
    retriever = build_retriever(load_chunks)

    queries = []
    labels = []

    for i in range(0, len(load_chunks), 2):
        query = (
            load_chunks["text"][i]
            + " "
            + subsample_query(load_chunks["text"][i + 1], k=15)
        )
        queries.append(query)
        labels.append([load_chunks["id"][i], load_chunks["id"][i + 1]])

    for query, label in zip(queries, labels):
        results = retriever.search([query], top_k=2)[0]

        assert results[0][0] == label[0]
        assert results[1][0] == label[1]

    retriever.delete(list(range(0, len(load_chunks), 2)))

    for query, label in zip(queries, labels):
        results = retriever.search([query], top_k=2)[0]

        assert results[0][0] == label[1]


def get_supervised_samples(dataset):
    random_words = []
    for sample in dataset["text"]:
        words = sample.split()
        random_words.extend(random.choices(words, k=4))

    acronyms = dataset["text"].map(
        lambda s: "".join(w[0] for w in s.split())
        + " "
        + " ".join(random.choices(random_words, k=5))
    )
    ids = dataset["id"]

    return ids, acronyms


def test_finetunable_retriever_finetuning(load_chunks):
    retriever = build_retriever(load_chunks)

    ids, acronyms = get_supervised_samples(load_chunks)

    acc_before_finetuning = compute_accuracy(retriever, queries=acronyms, labels=ids)
    print("before finetuning: p@1 =", acc_before_finetuning)
    assert acc_before_finetuning <= 0.5

    batches = []
    batch_size = 25
    for i in range(0, len(ids), batch_size):
        batches.append(
            SupervisedBatch(
                query=pd.Series(acronyms[i : i + batch_size]),
                chunk_id=pd.Series(map(lambda id: [id], ids[i : i + batch_size])),
            )
        )

    retriever.supervised_train(batches)

    acc_after_finetuning = compute_accuracy(retriever, queries=acronyms, labels=ids)
    print("after finetuning: p@1 =", acc_after_finetuning)
    assert acc_after_finetuning >= 0.9

    model_path = "ndb_finetunable_retriever_for_test"
    retriever.save(model_path)
    retriever = FinetunableRetriever.load(model_path)

    after_load_accuracy = compute_accuracy(retriever, acronyms, load_chunks["id"])

    assert acc_after_finetuning == after_load_accuracy

    os.remove(model_path)


def test_finetunable_retriever_upvote(load_chunks):
    retriever = build_retriever(load_chunks)

    ids, acronyms = get_supervised_samples(load_chunks)

    acc_before_upvote = compute_accuracy(retriever, queries=acronyms, labels=ids)
    print("before upvote p@1 =", acc_before_upvote)
    assert acc_before_upvote <= 0.5

    retriever.upvote(queries=acronyms, chunk_ids=ids)

    acc_after_upvote = compute_accuracy(retriever, queries=acronyms, labels=ids)
    print("after upvote p@1 =", acc_after_upvote)
    assert acc_after_upvote >= 0.9


def get_association_samples(dataset):

    ids, acronyms = get_supervised_samples(dataset)

    targets = (
        dataset["text"]
        .map(lambda s: " ".join(random.choices(s.split(), k=10)))
        .to_list()
    )

    return ids, acronyms, targets


def test_finetunable_retriever_associate(load_chunks):
    retriever = build_retriever(load_chunks)

    ids, acronyms, targets = get_association_samples(load_chunks)

    acc_before_associate = compute_accuracy(retriever, queries=acronyms, labels=ids)
    print("before associate p@1 =", acc_before_associate)
    assert acc_before_associate <= 0.5

    retriever.associate(sources=acronyms, targets=targets, associate_strength=1)

    acc_after_associate = compute_accuracy(retriever, queries=acronyms, labels=ids)
    print("after associate p@1 =", acc_after_associate)
    assert acc_after_associate >= 0.9
