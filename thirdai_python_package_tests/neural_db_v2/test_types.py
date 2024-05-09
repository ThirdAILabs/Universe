import pandas as pd
import pytest
from thirdai.neural_db_v2.core.types import (
    ChunkBatch,
    CustomIdSupervisedBatch,
    NewChunkBatch,
    SupervisedBatch,
)

pytestmark = [pytest.mark.unit]


def new_chunk_batch(use_custom_id=True, use_metadata=True, uneven_lengths=False):
    return NewChunkBatch(
        custom_id=pd.Series([1, 4]) if use_custom_id else None,
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
            custom_id=None,
            text=pd.Series(),
            keywords=pd.Series(),
            document=pd.Series(),
            metadata=None,
        )


@pytest.mark.parametrize("use_custom_id", [True, False])
def test_new_chunk_batch_no_custom_id(use_custom_id):
    new_chunk_batch(use_custom_id=use_custom_id)


@pytest.mark.parametrize("use_metadata", [True, False])
def test_new_chunk_batch_metadata(use_metadata):
    new_chunk_batch(use_metadata=use_metadata)


@pytest.mark.parametrize("use_custom_id", [True, False])
@pytest.mark.parametrize("use_metadata", [True, False])
def test_new_chunk_batch_get(use_custom_id, use_metadata):
    batch = new_chunk_batch(use_custom_id=use_custom_id, use_metadata=use_metadata)

    assert (batch.custom_id is not None) == use_custom_id
    if batch.custom_id is not None:
        assert batch[0].custom_id == 1

    assert (batch.metadata is not None) == use_metadata
    if batch.metadata is not None:
        assert batch[0].metadata == {"class": "a", "number": 4, "item": "x"}

    assert batch[0].text == "some text"
    assert batch[0].keywords == "important"
    assert batch[0].document == "doc0"


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


def test_custom_id_supervised_batch_uneven_lengths():
    with pytest.raises(
        ValueError,
        match="Must have fields of the same length in CustomIdSupervisedBatch.",
    ):
        CustomIdSupervisedBatch(
            query=pd.Series(["l"]),
            custom_id=pd.Series([0, 1]),
        )


def test_custom_id_supervised_batch_empty():
    with pytest.raises(
        ValueError,
        match="Cannot create empty CustomIdSupervisedBatch.",
    ):
        CustomIdSupervisedBatch(
            query=pd.Series(),
            custom_id=pd.Series(),
        )
