import pandas as pd
import pytest
from thirdai.neural_db_v2.core.types import ChunkBatch, NewChunkBatch, SupervisedBatch

pytestmark = [pytest.mark.unit]


def new_chunk_batch(use_metadata=True, uneven_lengths=False):
    return NewChunkBatch(
        text=pd.Series(["some text", "more text"]),
        keywords=(
            pd.Series(["only one"])
            if uneven_lengths
            else pd.Series(["important", "key word"])
        ),
        document=pd.Series(["doc0", "doc1"]),
        metadata=(
            pd.DataFrame({"class": ["a", "b"], "number": [4, 9], "item": ["x", "y"]})
            if use_metadata
            else None
        ),
    )


def test_new_chunk_batch_uneven_lengths():
    with pytest.raises(
        ValueError,
        match="Must have fields of the same length in NewChunkBatch.",
    ):
        new_chunk_batch(uneven_lengths=True)


def test_new_chunk_batch_empty():
    with pytest.raises(
        ValueError,
        match="Cannot create empty NewChunkBatch.",
    ):
        NewChunkBatch(
            text=pd.Series(),
            keywords=pd.Series(),
            document=pd.Series(),
            metadata=None,
        )


@pytest.mark.parametrize("use_metadata", [True, False])
def test_new_chunk_batch_metadata(use_metadata):
    new_chunk_batch(use_metadata=use_metadata)


def test_chunk_batch_uneven_lengths():
    with pytest.raises(
        ValueError,
        match="Must have fields of the same length in ChunkBatch.",
    ):
        ChunkBatch(
            text=pd.Series(["l"]),
            keywords=pd.Series(["p"]),
            chunk_id=pd.Series([0, 1]),
        )


def test_chunk_batch_empty():
    with pytest.raises(
        ValueError,
        match="Cannot create empty ChunkBatch.",
    ):
        ChunkBatch(
            text=pd.Series(),
            keywords=pd.Series(),
            chunk_id=pd.Series(),
        )


def test_supervised_batch_uneven_lengths():
    with pytest.raises(
        ValueError,
        match="Must have fields of the same length in SupervisedBatch.",
    ):
        SupervisedBatch(
            query=pd.Series(["l"]),
            chunk_id=pd.Series([1, 2]),
        )


def test_supervised_batch_empty():
    with pytest.raises(
        ValueError,
        match="Cannot create empty SupervisedBatch.",
    ):
        SupervisedBatch(
            query=pd.Series(),
            chunk_id=pd.Series(),
        )
