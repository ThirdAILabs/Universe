import os

import pandas as pd
import pytest
from thirdai.neural_db_v2.core.types import ChunkBatch, SupervisedBatch
from thirdai.neural_db_v2.retrievers.mach import Mach

pytestmark = [pytest.mark.release]


@pytest.fixture(scope="session")
def chunks():
    filename = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../auto_ml/python_tests/texts.csv",
    )
    return pd.read_csv(filename)


def build_retriever(chunk_df):
    chunk_batches = []
    batch_size = 5
    for i in range(0, len(chunk_df), batch_size):
        chunks = chunk_df.iloc[i : i + batch_size]
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


def test_ndb_mach_retriever_search(chunks):
    retriever = build_retriever(chunks)

    n = len(chunks)
    for _, row in chunks.iterrows():
        id = row["id"]
        search_results = retriever.search([row["text"]], top_k=1)
        assert id == search_results[0][0][0]
        rank_results = retriever.rank(
            [row["text"]], choices=[set([id, (id + 1) % n])], top_k=1
        )
        assert id == rank_results[0][0][0]


def test_ndb_mach_retriever_supervised_train(chunks):
    retriever = build_retriever(chunks)

    supervised_batch = SupervisedBatch(
        query=[str(chunk_id) for chunk_id in chunks["id"]], chunk_id=chunks["id"]
    )

    retriever.supervised_train([supervised_batch], epochs=10)

    accuracy = 0
    for i in range(len(chunks)):
        results = retriever.search([str(i)], top_k=1)
        if results[0][0][0] == i:
            accuracy += 1

    assert accuracy > 0.8
