import pandas as pd
import pytest
from ndbv2_test_utils import simple_chunks_df
from thirdai.neural_db_v2.core.types import ChunkBatch, SupervisedBatch
from thirdai.neural_db_v2.retrievers.mach import Mach

pytestmark = [pytest.mark.release]


@pytest.fixture(scope="session")
def build_retriever(simple_chunks_df):
    chunk_batches = []
    batch_size = 5
    for i in range(0, len(simple_chunks_df), batch_size):
        chunks = simple_chunks_df.iloc[i : i + batch_size]
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


def test_ndb_mach_retriever_search(build_retriever, simple_chunks_df):
    retriever = build_retriever

    n = len(simple_chunks_df)
    search_accuracy = 0
    rank_accuracy = 0
    for _, row in simple_chunks_df.iterrows():
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


def get_accuracy(retriever, queries, ids):
    accuracy = 0
    for query, id in zip(queries, ids):
        results = retriever.search([query], top_k=1)
        if results[0][0][0] == id:
            accuracy += 1

    return accuracy / len(queries)


def test_ndb_mach_retriever_supervised_train(build_retriever, simple_chunks_df):
    retriever = build_retriever

    queries = [str(chunk_id) for chunk_id in simple_chunks_df["id"]]

    supervised_batch = SupervisedBatch(query=queries, chunk_id=simple_chunks_df["id"])

    before_accuracy = get_accuracy(retriever, queries, simple_chunks_df["id"])
    assert before_accuracy < 0.5

    retriever.supervised_train([supervised_batch], epochs=15, learning_rate=0.1)

    after_accuracy = get_accuracy(retriever, queries, simple_chunks_df["id"])
    assert after_accuracy > 0.9


def test_ndb_mach_retriever_delete(build_retriever, simple_chunks_df):
    retriever = build_retriever

    before_del_results = retriever.search([simple_chunks_df["text"][0]], top_k=1)
    retriever.delete([before_del_results[0][0][0]])
    after_del_results = retriever.search([simple_chunks_df["text"][0]], top_k=1)

    assert before_del_results[0][0][0] != after_del_results[0][0][0]
