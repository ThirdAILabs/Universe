import pandas as pd
import pytest
from ndbv2_test_utils import simple_chunks_df
from thirdai.neural_db_v2.core.types import ChunkBatch
from thirdai.neural_db_v2.retrievers.inverted_index import InvertedIndex

pytestmark = [pytest.mark.release]


def build_index(chunk_df):
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

    index = InvertedIndex()

    index.insert(chunk_batches)

    return index


def test_inverted_index_retriever_search(simple_chunks_df):
    index = build_index(simple_chunks_df)

    for _, row in simple_chunks_df.iterrows():
        id = row["id"]
        search_results = index.search([row["text"]], top_k=1)
        assert id == search_results[0][0][0]
        rank_results = index.rank(
            [row["text"]], choices=[set([id + 2, id, id + 1])], top_k=1
        )
        assert id == rank_results[0][0][0]
