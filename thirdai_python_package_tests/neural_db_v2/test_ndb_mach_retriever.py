import os

import pandas as pd
import pytest
from ndbv2_utils import compute_accuracy, load_chunks
from thirdai.neural_db_v2.core.types import ChunkBatch, SupervisedBatch
from thirdai.neural_db_v2.retrievers import Mach

pytestmark = [pytest.mark.release]


@pytest.fixture(scope="session")
def build_retriever(load_chunks):
    chunk_batches = []
    batch_size = 5
    for i in range(0, len(load_chunks), batch_size):
        chunks = load_chunks.iloc[i : i + batch_size]
        chunk_batches.append(
            ChunkBatch(
                text=chunks["text"],
                keywords=pd.Series(["" for _ in range(len(chunks))]),
                chunk_id=chunks["id"],
            )
        )

    retriever = Mach()

    retriever.insert(chunk_batches)

    return retriever


def test_ndb_mach_retriever_search(build_retriever, load_chunks):
    retriever = build_retriever

    n = len(load_chunks)
    search_accuracy = 0
    rank_accuracy = 0
    for _, row in load_chunks.iterrows():
        id = row["id"]
        search_results = retriever.search([row["text"]], top_k=1)
        if id == search_results[0][0][0]:
            search_accuracy += 1
        rank_results = retriever.rank(
            [row["text"]], choices=[set([id, (id + 1) % n])], top_k=1
        )
        if id == rank_results[0][0][0]:
            rank_accuracy += 1

    assert search_accuracy / n > 0.9
    assert rank_accuracy / n > 0.9


def test_ndb_mach_retriever_supervised_train(build_retriever, load_chunks):
    retriever = build_retriever

    queries = [str(chunk_id) for chunk_id in load_chunks["id"]]

    supervised_batch = SupervisedBatch(
        query=queries, chunk_id=load_chunks["id"].map(lambda id: [id])
    )

    before_sup_accuracy = compute_accuracy(retriever, queries, load_chunks["id"])
    assert before_sup_accuracy < 0.5

    retriever.supervised_train([supervised_batch], epochs=15, learning_rate=0.1)

    after_sup_accuracy = compute_accuracy(retriever, queries, load_chunks["id"])
    assert after_sup_accuracy > 0.9

    model_path = "ndb_mach_retriever_for_test"
    retriever.save(model_path)
    retriever = Mach.load(model_path)

    after_load_accuracy = compute_accuracy(retriever, queries, load_chunks["id"])

    assert after_sup_accuracy == after_load_accuracy

    os.remove(model_path)


def test_ndb_mach_retriever_delete(build_retriever, load_chunks):
    retriever = build_retriever

    before_del_results = retriever.search([load_chunks["text"][0]], top_k=1)
    retriever.delete([before_del_results[0][0][0]])
    after_del_results = retriever.search([load_chunks["text"][0]], top_k=1)

    assert before_del_results[0][0][0] != after_del_results[0][0][0]
