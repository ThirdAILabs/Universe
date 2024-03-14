from thirdai.neural_db_v2.chunk_stores.sqlite_chunk_store import SQLiteChunkStore
from thirdai.neural_db_v2.core.types import NewChunkBatch, CustomIdSupervisedBatch
import pytest
import pandas as pd

pytestmark = [pytest.mark.release]


def get_simple_chunk_store():
    store = SQLiteChunkStore()

    batches = [
        NewChunkBatch(
            custom_id=None,
            text=pd.Series(["0 1", "1 2"]),
            keywords=pd.Series(["00 01", "10 11"]),
            metadata=None,
            document=pd.Series(["doc0", "doc1"]),
        ),
        NewChunkBatch(
            custom_id=pd.Series([200, 300, 400]),
            text=pd.Series(["2 3", "3 4", "4 5"]),
            keywords=pd.Series(["20 21", "30 31", "40, 41"]),
            metadata=None,
            document=pd.Series(["doc2", "doc3", "doc4"]),
        ),
    ]

    inserted_batches = store.insert(batches)

    assert len(inserted_batches) == 2

    assert (inserted_batches[0].chunk_id == pd.Series([0, 1])).all()
    assert (inserted_batches[1].chunk_id == pd.Series([2, 3, 4])).all()

    return store


def test_sqlite_chunk_store_basic_operations():
    store = get_simple_chunk_store()

    chunks = store.get_chunks([1, 3, 0])
    assert len(chunks) == 3
    for chunk, i in zip(chunks, [1, 3, 0]):
        assert chunk.chunk_id == i
        assert chunk.text == f"{i} {i+1}"
        assert chunk.keywords == f"{i}{0} {i}{1}"
        assert chunk.document == f"doc{i}"

    store.delete([1, 2])

    with pytest.raises(ValueError, match="Could not find chunk with id 1."):
        store.get_chunks([3, 1, 0])


def test_sqlite_chunk_store_remapping():
    store = get_simple_chunk_store()

    remapped_batch = store.remap_custom_ids(
        [
            CustomIdSupervisedBatch(
                query=pd.Series(["w", "x", "y", "z"]),
                custom_id=pd.Series([[200], [400, 300], [300], [300, 200]]),
            )
        ]
    )[0]

    assert (remapped_batch.chunk_id == pd.Series([[2], [4, 3], [3], [3, 2]])).all()
