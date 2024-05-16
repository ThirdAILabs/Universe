import pandas as pd
import pytest
from ndbv2_utils import load_chunks
from thirdai.neural_db_v2.core.types import ChunkBatch, SupervisedBatch
from thirdai.neural_db_v2.retrievers.mach import Mach
from thirdai.neural_db_v2.retrievers.mach_ensemble import MachEnsemble

pytestmark = [pytest.mark.release]


@pytest.fixture(scope="session")
def build_mach_retriever(load_chunks):
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


@pytest.fixture(scope="session")
def build_mach_ensemble(load_chunks):
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

    retriever = MachEnsemble(n_models=5, n_hashes=1, n_buckets=10_000)

    retriever.insert(chunk_batches)

    return retriever


def check_mach_search(retriever, load_chunks):
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


def test_ndb_mach_retriever_search(build_mach_retriever, load_chunks):
    check_mach_search(retriever=build_mach_retriever, load_chunks=load_chunks)


def test_ndb_mach_ensemble_search(build_mach_ensemble, load_chunks):
    check_mach_search(retriever=build_mach_ensemble, load_chunks=load_chunks)


def get_accuracy(retriever, queries, ids):
    accuracy = 0
    for query, id in zip(queries, ids):
        results = retriever.search([query], top_k=1)
        if results[0][0][0] == id:
            accuracy += 1

    return accuracy / len(queries)


def check_mach_supervised_train(retriever, load_chunks):
    queries = [str(chunk_id) for chunk_id in load_chunks["id"]]

    supervised_batch = SupervisedBatch(query=queries, chunk_id=load_chunks["id"])

    before_accuracy = get_accuracy(retriever, queries, load_chunks["id"])
    assert before_accuracy < 0.5

    retriever.supervised_train([supervised_batch], epochs=15, learning_rate=0.1)

    after_accuracy = get_accuracy(retriever, queries, load_chunks["id"])
    assert after_accuracy > 0.9


def test_ndb_mach_retriever_supervised_train(build_mach_retriever, load_chunks):
    check_mach_supervised_train(retriever=build_mach_retriever, load_chunks=load_chunks)


def test_ndb_mach_ensemble_supervised_train(build_mach_ensemble, load_chunks):
    check_mach_supervised_train(retriever=build_mach_ensemble, load_chunks=load_chunks)


def check_mach_delete(retriever, load_chunks):
    before_del_results = retriever.search([load_chunks["text"][0]], top_k=1)
    retriever.delete([before_del_results[0][0][0]])
    after_del_results = retriever.search([load_chunks["text"][0]], top_k=1)

    assert before_del_results[0][0][0] != after_del_results[0][0][0]


def test_ndb_mach_retriever_delete(build_mach_retriever, load_chunks):
    check_mach_delete(retriever=build_mach_retriever, load_chunks=load_chunks)


def test_ndb_mach_ensemble_delete(build_mach_ensemble, load_chunks):
    check_mach_delete(retriever=build_mach_ensemble, load_chunks=load_chunks)
