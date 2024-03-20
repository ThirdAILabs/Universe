import os

import pandas as pd
import pytest
from thirdai.neural_db_v2.core.types import ChunkBatch
from thirdai.neural_db_v2.retrievers.inverted_index import InvertedIndex

pytestmark = [pytest.mark.release]


@pytest.fixture(scope="session")
def load_chunks():
    filename = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../auto_ml/python_tests/texts.csv",
    )

    return pd.read_csv(filename)


def build_index(chunk_df):
    chunk_batches = []
    batch_size = 5
    for i in range(0, len(chunk_df), batch_size):
        chunks = chunk_df.iloc[i : i + batch_size]
        chunk_batches.append(
            ChunkBatch(
                custom_id=None,
                text=chunks["text"],
                keywords=pd.Series(["" for _ in range(len(chunks))]),
                metadata=None,
                document=pd.Series(["texts.csv" for _ in range(len(chunks))]),
                chunk_id=chunks["id"],
            )
        )

    index = InvertedIndex()

    index.insert(chunk_batches)

    return index


def test_inverted_index_search(load_chunks):
    index = build_index(load_chunks)

    for _, row in load_chunks.iterrows():
        id = row["id"]
        search_results = index.search([row["text"]], top_k=1)
        assert id == search_results[0][0][0]
        rank_results = index.rank(
            [row["text"]], choices=[[id + 2, id, id + 1]], top_k=1
        )
        assert id == rank_results[0][0][0]
