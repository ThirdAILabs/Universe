import pandas as pd
import pytest
from thirdai.neural_db_v2.chunk_stores.constraints import (
    AnyOf,
    EqualTo,
    GreaterThan,
    LessThan,
)
from thirdai.neural_db_v2.chunk_stores.pandas_chunk_store import PandasChunkStore
from thirdai.neural_db_v2.chunk_stores.sqlite_chunk_store import SQLiteChunkStore
from thirdai.neural_db_v2.core.types import CustomIdSupervisedBatch, NewChunkBatch

pytestmark = [pytest.mark.release]


def assert_chunk_ids(inserted_batches, expected_chunk_ids):
    actual_chunk_ids = []
    for batch in inserted_batches:
        for chunk_id in batch.chunk_id:
            actual_chunk_ids.append(chunk_id)

    assert expected_chunk_ids == actual_chunk_ids


def get_simple_chunk_store(chunk_store_type, custom_id_type=int):
    store = chunk_store_type()

    batches = [
        NewChunkBatch(
            custom_id=None,
            text=pd.Series(["0 1", "1 2"]),
            keywords=pd.Series(["00 11", "11 22"]),
            document=pd.Series(["doc0", "doc1"]),
            metadata=pd.DataFrame(
                {"class": ["a", "b"], "number": [4, 9], "item": ["x", "y"]}
            ),
        ),
        NewChunkBatch(
            custom_id=pd.Series(
                [custom_id_type(200), custom_id_type(300), custom_id_type(400)]
            ),
            text=pd.Series(["2 3", "3 4", "4 5"]),
            keywords=pd.Series(["22 33", "33 44", "44, 55"]),
            document=pd.Series(["doc2", "doc3", "doc4"]),
            metadata=pd.DataFrame(
                {"class": ["c", "b", "a"], "number": [7, 2, 4], "time": [1.4, 2.6, 3.4]}
            ),
        ),
    ]

    inserted_batches = store.insert(batches)

    assert_chunk_ids(inserted_batches, [0, 1, 2, 3, 4])

    return store


@pytest.mark.parametrize("chunk_store", [SQLiteChunkStore, PandasChunkStore])
def test_chunk_store_basic_operations(chunk_store):
    def check_chunk_contents(chunk, chunk_id, value, custom_id=None, metadata=None):
        assert chunk.chunk_id == chunk_id
        assert chunk.text == f"{value} {value+1}"
        assert chunk.keywords == f"{value}{value} {value+1}{value+1}"
        assert chunk.document == f"doc{value}"
        assert chunk.custom_id == custom_id
        assert chunk.metadata == metadata

    store = get_simple_chunk_store(chunk_store)

    chunks = store.get_chunks([1, 3, 0])
    assert len(chunks) == 3
    check_chunk_contents(
        chunks[0],
        chunk_id=1,
        value=1,
        metadata={"class": "b", "number": 9, "item": "y", "time": None},
    )
    check_chunk_contents(
        chunks[1],
        chunk_id=3,
        value=3,
        custom_id=300,
        metadata={"class": "b", "number": 2, "time": 2.6, "item": None},
    )
    check_chunk_contents(
        chunks[2],
        chunk_id=0,
        value=0,
        metadata={"class": "a", "number": 4, "item": "x", "time": None},
    )

    store.delete([1, 2])

    with pytest.raises(ValueError, match="Could not find chunk with .*"):
        store.get_chunks([3, 1, 0])

    chunks = store.get_chunks([3, 0])
    assert len(chunks) == 2
    check_chunk_contents(
        chunks[0],
        chunk_id=3,
        value=3,
        custom_id=300,
        metadata={"class": "b", "number": 2, "time": 2.6, "item": None},
    )
    check_chunk_contents(
        chunks[1],
        chunk_id=0,
        value=0,
        metadata={"class": "a", "number": 4, "item": "x", "time": None},
    )

    new_batches = [
        NewChunkBatch(
            custom_id=pd.Series([1000, 2000]),
            text=pd.Series(["7 8", "1 2"]),
            keywords=pd.Series(["77 88", "11 22"]),
            document=pd.Series(["doc7", "doc1"]),
            metadata=pd.DataFrame(
                {"class": ["c", "d"], "time": [7.2, 8.1], "item": ["y", "z"]}
            ),
        ),
    ]

    inserted_batches = store.insert(new_batches)
    assert_chunk_ids(inserted_batches, [5, 6])

    chunks = store.get_chunks([6, 0, 5])
    check_chunk_contents(
        chunks[0],
        chunk_id=6,
        value=1,
        custom_id=2000,
        metadata={"class": "d", "number": None, "time": 8.1, "item": "z"},
    )
    check_chunk_contents(
        chunks[1],
        chunk_id=0,
        value=0,
        metadata={"class": "a", "number": 4, "item": "x", "time": None},
    )
    check_chunk_contents(
        chunks[2],
        chunk_id=5,
        value=7,
        custom_id=1000,
        metadata={"class": "c", "number": None, "time": 7.2, "item": "y"},
    )


def test_sqlite_chunk_store_custom_id_type_mismatch():
    integer_label_batch = NewChunkBatch(
        custom_id=pd.Series([200]),
        text=pd.Series(["2 3"]),
        keywords=pd.Series(["20 21"]),
        document=pd.Series(["doc2"]),
        metadata=None,
    )

    string_label_batch = NewChunkBatch(
        custom_id=pd.Series(["apple"]),
        text=pd.Series(["0 1"]),
        keywords=pd.Series(["00 01"]),
        document=pd.Series(["doc0"]),
        metadata=None,
    )

    with pytest.raises(
        ValueError,
        match="Custom ids must all have the same type. Found some custom ids with type int, and some with type str.",
    ):
        store = SQLiteChunkStore()
        store.insert(
            chunks=[
                integer_label_batch,
                string_label_batch,
            ]
        )

    with pytest.raises(
        ValueError,
        match="Custom ids must all have the same type. Found some custom ids with type int, and some with type str.",
    ):
        store = SQLiteChunkStore()
        store.insert(
            chunks=[
                string_label_batch,
                integer_label_batch,
            ]
        )


@pytest.mark.parametrize("chunk_store", [SQLiteChunkStore, PandasChunkStore])
def test_chunk_store_constraints_equal_to(chunk_store):
    store = get_simple_chunk_store(chunk_store)

    chunk_ids = store.filter_chunk_ids({"class": EqualTo("b")})
    assert chunk_ids == set([1, 3])


@pytest.mark.parametrize("chunk_store", [SQLiteChunkStore, PandasChunkStore])
def test_chunk_store_constraints_any_of(chunk_store):
    store = get_simple_chunk_store(chunk_store)

    chunk_ids = store.filter_chunk_ids({"number": AnyOf([4, 2])})
    assert chunk_ids == set([0, 3, 4])


@pytest.mark.parametrize("chunk_store", [SQLiteChunkStore, PandasChunkStore])
def test_chunk_store_constraints_greater_than(chunk_store):
    store = get_simple_chunk_store(chunk_store)

    chunk_ids = store.filter_chunk_ids({"number": GreaterThan(7, inclusive=True)})
    assert chunk_ids == set([1, 2])

    chunk_ids = store.filter_chunk_ids({"number": GreaterThan(7, inclusive=False)})
    assert chunk_ids == set([1])


@pytest.mark.parametrize("chunk_store", [SQLiteChunkStore, PandasChunkStore])
def test_chunk_store_constraints_less_than(chunk_store):
    store = get_simple_chunk_store(chunk_store)

    chunk_ids = store.filter_chunk_ids({"number": LessThan(4, inclusive=True)})
    assert chunk_ids == set([0, 3, 4])

    chunk_ids = store.filter_chunk_ids({"number": LessThan(4, inclusive=False)})
    assert chunk_ids == set([3])


@pytest.mark.parametrize("chunk_store", [SQLiteChunkStore, PandasChunkStore])
def test_chunk_store_constraints_multiple_constraints(chunk_store):
    store = get_simple_chunk_store(chunk_store)

    chunk_ids = store.filter_chunk_ids(
        {
            "number": GreaterThan(6, inclusive=True),
            "class": AnyOf(["b", "c"]),
        }
    )
    assert chunk_ids == set([1, 2])

    chunk_ids = store.filter_chunk_ids(
        {
            "number": GreaterThan(6, inclusive=True),
            "class": AnyOf(["b", "c"]),
            "time": LessThan(3),
        }
    )
    assert chunk_ids == set([2])

    chunk_ids = store.filter_chunk_ids(
        {
            "number": GreaterThan(6, inclusive=True),
            "class": AnyOf(["b", "c"]),
            "item": EqualTo("y"),
        }
    )
    assert chunk_ids == set([1])


@pytest.mark.parametrize("chunk_store", [SQLiteChunkStore, PandasChunkStore])
def test_chunk_store_constraints_return_valid_ids(chunk_store):
    store = get_simple_chunk_store(chunk_store)

    chunk_ids = store.filter_chunk_ids({"number": LessThan(4, inclusive=True)})
    assert chunk_ids == set([0, 3, 4])

    store.delete([3])

    chunk_ids = store.filter_chunk_ids({"number": LessThan(4, inclusive=True)})
    assert chunk_ids == set([0, 4])


@pytest.mark.parametrize("chunk_store", [SQLiteChunkStore, PandasChunkStore])
@pytest.mark.parametrize("id_type", [int, str])
def test_chunk_store_remapping(chunk_store, id_type):
    store = get_simple_chunk_store(chunk_store, custom_id_type=id_type)

    remapped_batch = store.remap_custom_ids(
        [
            CustomIdSupervisedBatch(
                query=pd.Series(["w", "x", "y", "z"]),
                custom_id=pd.Series(
                    [
                        [id_type(200)],
                        [id_type(400), id_type(300)],
                        [id_type(300)],
                        [id_type(300), id_type(200)],
                    ]
                ),
            )
        ]
    )[0]

    assert (remapped_batch.chunk_id == pd.Series([[2], [4, 3], [3], [3, 2]])).all()

    with pytest.raises(
        ValueError, match=f"Could not find chunk with custom id {id_type(700)}."
    ):
        store.remap_custom_ids(
            [
                CustomIdSupervisedBatch(
                    query=pd.Series(["w", "x"]),
                    custom_id=pd.Series([[id_type(200)], [id_type(400), id_type(700)]]),
                )
            ]
        )


def test_sql_lite_chunk_store_batching():
    store = SQLiteChunkStore()

    new_batch = NewChunkBatch(
        custom_id=None,
        text=pd.Series(["0", "1", "2"]),
        keywords=pd.Series(["00 11", "11 22", "22 33"]),
        document=pd.Series(["doc0", "doc1", "doc2"]),
        metadata=pd.DataFrame(
            {"class": ["a", "b", "c"], "number": [4, 9, 11], "item": ["x", "y", "z"]}
        ),
    )

    inserted_batches_1 = store.insert([new_batch], sql_lite_iterator_batch_size=2)
    inserted_batches_2 = store.insert([new_batch], sql_lite_iterator_batch_size=2)

    def assert_lens(inserted_batches):
        num_batches = 0
        num_rows = 0
        for batch in inserted_batches:
            num_batches += 1
            num_rows += len(batch.text)

        assert num_batches == 2
        assert num_rows == 3

    # we check this out of order to verify that the min/max chunk_id logic works
    # in the SQLLiteIterator object, ie that we pull the right range of chunk_ids
    # despite using the iterators out of order
    assert_lens(inserted_batches_2)
    assert_lens(inserted_batches_1)
