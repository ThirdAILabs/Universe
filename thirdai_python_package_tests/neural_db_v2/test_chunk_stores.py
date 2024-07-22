import itertools
import os

import pandas as pd
import pytest
from ndbv2_utils import clean_up_sql_lite_db
from thirdai.neural_db_v2.chunk_stores import PandasChunkStore, SQLiteChunkStore
from thirdai.neural_db_v2.chunk_stores.constraints import (
    AnyOf,
    EqualTo,
    GreaterThan,
    LessThan,
)
from thirdai.neural_db_v2.core.types import VersionedNewChunkBatch

pytestmark = [pytest.mark.release]


def assert_chunk_ids(inserted_batches, expected_chunk_ids):
    actual_chunk_ids = []
    for batch in inserted_batches:
        for chunk_id in batch.chunk_id:
            actual_chunk_ids.append(chunk_id)

    assert expected_chunk_ids == actual_chunk_ids


def get_simple_chunk_store(chunk_store_type, use_metadata=True):
    store = chunk_store_type()

    batches = [
        [
            VersionedNewChunkBatch(
                text=pd.Series(["0 1", "1 2"]),
                keywords=pd.Series(["00 11", "11 22"]),
                document=pd.Series(["doc0", "doc1"]),
                metadata=(
                    pd.DataFrame(
                        {"class": ["a", "b"], "number": [4, 9], "item": ["x", "y"]}
                    )
                    if use_metadata
                    else None
                ),
                doc_id=pd.Series(["00", "11"]),
                doc_version=pd.Series([1, 1]),
            ),
            VersionedNewChunkBatch(
                text=pd.Series(["2 3", "3 4", "4 5"]),
                keywords=pd.Series(["22 33", "33 44", "44, 55"]),
                document=pd.Series(["doc2", "doc3", "doc4"]),
                metadata=(
                    pd.DataFrame(
                        {
                            "class": ["c", "b", "a"],
                            "number": [7, 2, 4],
                            "time": [1.4, 2.6, 3.4],
                        }
                    )
                    if use_metadata
                    else None
                ),
                doc_id=pd.Series(["22", "33", "44"]),
                doc_version=pd.Series([1, 1, 1]),
            ),
        ],
        [
            VersionedNewChunkBatch(
                text=pd.Series(["5 6", "7 8"]),
                keywords=pd.Series(["55 66", "77 88"]),
                document=pd.Series(["doc5", "doc6"]),
                metadata=(
                    pd.DataFrame(
                        {"class": ["x", "y"], "time": [6.5, 2.4], "item": ["t", "u"]}
                    )
                    if use_metadata
                    else None
                ),
                doc_id=pd.Series(["55", "66"]),
                doc_version=pd.Series([1, 1]),
            ),
        ],
    ]

    inserted_batches, chunk_ids = store.insert(batches)

    assert len(batches) == len(chunk_ids)

    for doc_batches, doc_chunk_ids in zip(batches, chunk_ids):
        index = 0
        for batch in doc_batches:
            for i in range(len(batch)):
                chunk = store.get_chunks([doc_chunk_ids.chunk_ids[index]])[0]

                assert batch.text[i] == chunk.text
                assert batch.keywords[i] == chunk.keywords
                assert batch.document[i] == chunk.document

                index += 1

    assert_chunk_ids(inserted_batches, [0, 1, 2, 3, 4, 5, 6])

    return store


def check_chunk_contents(chunk, chunk_id, value, metadata=None):
    assert chunk.chunk_id == chunk_id
    assert chunk.text == f"{value} {value+1}"
    assert chunk.keywords == f"{value}{value} {value+1}{value+1}"
    assert chunk.document == f"doc{value}"
    assert chunk.metadata == metadata
    assert chunk.doc_id == f"{value}{value}"
    assert chunk.doc_version == 1


@pytest.mark.parametrize("chunk_store", [SQLiteChunkStore, PandasChunkStore])
def test_chunk_store_basic_operations(chunk_store):
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
        metadata={"class": "b", "number": 2, "time": 2.6, "item": None},
    )
    check_chunk_contents(
        chunks[1],
        chunk_id=0,
        value=0,
        metadata={"class": "a", "number": 4, "item": "x", "time": None},
    )

    new_batches = [
        VersionedNewChunkBatch(
            text=pd.Series(["8 9", "1 2"]),
            keywords=pd.Series(["88 99", "11 22"]),
            document=pd.Series(["doc8", "doc1"]),
            metadata=pd.DataFrame(
                {"class": ["c", "d"], "time": [7.2, 8.1], "item": ["y", "z"]}
            ),
            doc_id=pd.Series(["88", "11"]),
            doc_version=pd.Series([1, 1]),
        ),
    ]

    inserted_batches, _ = store.insert([new_batches])
    assert_chunk_ids(inserted_batches, [7, 8])

    chunks = store.get_chunks([8, 0, 7])
    check_chunk_contents(
        chunks[0],
        chunk_id=8,
        value=1,
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
        chunk_id=7,
        value=8,
        metadata={"class": "c", "number": None, "time": 7.2, "item": "y"},
    )

    path = "test_chunk_store_basic_operations.store"
    store.save(path)
    clean_up_sql_lite_db(store)
    store = chunk_store.load(path)
    chunks = store.get_chunks([8, 0, 5])
    check_chunk_contents(
        chunks[0],
        chunk_id=8,
        value=1,
        metadata={"class": "d", "number": None, "time": 8.1, "item": "z"},
    )

    os.remove(path)


@pytest.mark.parametrize("chunk_store", [SQLiteChunkStore, PandasChunkStore])
def test_chunk_store_constraints_equal_to(chunk_store):
    store = get_simple_chunk_store(chunk_store)

    chunk_ids = store.filter_chunk_ids({"class": EqualTo("b")})
    assert chunk_ids == set([1, 3])

    clean_up_sql_lite_db(store)


@pytest.mark.parametrize("chunk_store", [SQLiteChunkStore, PandasChunkStore])
def test_chunk_store_constraints_any_of(chunk_store):
    store = get_simple_chunk_store(chunk_store)

    chunk_ids = store.filter_chunk_ids({"number": AnyOf([4, 2])})
    assert chunk_ids == set([0, 3, 4])

    clean_up_sql_lite_db(store)


@pytest.mark.parametrize("chunk_store", [SQLiteChunkStore, PandasChunkStore])
def test_chunk_store_constraints_greater_than(chunk_store):
    store = get_simple_chunk_store(chunk_store)

    chunk_ids = store.filter_chunk_ids({"number": GreaterThan(7, inclusive=True)})
    assert chunk_ids == set([1, 2])

    chunk_ids = store.filter_chunk_ids({"number": GreaterThan(7, inclusive=False)})
    assert chunk_ids == set([1])

    clean_up_sql_lite_db(store)


@pytest.mark.parametrize("chunk_store", [SQLiteChunkStore, PandasChunkStore])
def test_chunk_store_constraints_less_than(chunk_store):
    store = get_simple_chunk_store(chunk_store)

    chunk_ids = store.filter_chunk_ids({"number": LessThan(4, inclusive=True)})
    assert chunk_ids == set([0, 3, 4])

    chunk_ids = store.filter_chunk_ids({"number": LessThan(4, inclusive=False)})
    assert chunk_ids == set([3])

    clean_up_sql_lite_db(store)


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

    clean_up_sql_lite_db(store)


@pytest.mark.parametrize("chunk_store", [SQLiteChunkStore, PandasChunkStore])
def test_chunk_store_constraints_return_valid_ids(chunk_store):
    store = get_simple_chunk_store(chunk_store)

    chunk_ids = store.filter_chunk_ids({"number": LessThan(4, inclusive=True)})
    assert chunk_ids == set([0, 3, 4])

    store.delete([3])

    chunk_ids = store.filter_chunk_ids({"number": LessThan(4, inclusive=True)})
    assert chunk_ids == set([0, 4])

    clean_up_sql_lite_db(store)


def test_sql_lite_chunk_store_batching():
    store = SQLiteChunkStore()

    new_batch = VersionedNewChunkBatch(
        text=pd.Series(["0", "1", "2"]),
        keywords=pd.Series(["00 11", "11 22", "22 33"]),
        document=pd.Series(["doc0", "doc1", "doc2"]),
        metadata=pd.DataFrame(
            {"class": ["a", "b", "c"], "number": [4, 9, 11], "item": ["x", "y", "z"]}
        ),
        doc_id=pd.Series(["11", "22", "33"]),
        doc_version=pd.Series([1, 1, 1]),
    )

    inserted_batches_1, _ = store.insert([[new_batch]])
    inserted_batches_2, _ = store.insert([[new_batch]])

    def assert_lens(inserted_batches):
        num_batches = 0
        num_rows = 0
        for batch in inserted_batches:
            num_batches += 1
            num_rows += len(batch.text)

        assert num_batches == 1
        assert num_rows == 3

    # we check this out of order to verify that the min/max chunk_id logic works
    # in the SQLLiteIterator object, ie that we pull the right range of chunk_ids
    # despite using the iterators out of order
    assert_lens(inserted_batches_2)
    assert_lens(inserted_batches_1)

    clean_up_sql_lite_db(store)


@pytest.mark.parametrize("chunk_store", [SQLiteChunkStore, PandasChunkStore])
def test_chunk_store_with_no_metadata(chunk_store):
    store = get_simple_chunk_store(chunk_store, use_metadata=False)

    chunks = store.get_chunks([1, 3, 0])
    assert len(chunks) == 3
    check_chunk_contents(
        chunks[0],
        chunk_id=1,
        value=1,
        metadata=None,
    )
    check_chunk_contents(
        chunks[1],
        chunk_id=3,
        value=3,
        metadata=None,
    )
    check_chunk_contents(
        chunks[2],
        chunk_id=0,
        value=0,
        metadata=None,
    )

    store.delete([0])

    with pytest.raises(ValueError, match="Cannot filter constraints with no metadata."):
        store.filter_chunk_ids({"class": EqualTo("b")})

    clean_up_sql_lite_db(store)


@pytest.mark.parametrize("chunk_store", [SQLiteChunkStore, PandasChunkStore])
def test_chunk_store_max_version(chunk_store):
    store = get_simple_chunk_store(chunk_store, use_metadata=False)

    assert store.max_version_for_doc("new_id") == 0

    store.insert(
        [
            [
                VersionedNewChunkBatch(
                    text=pd.Series(["a b c"]),
                    keywords=pd.Series(["a b c"]),
                    metadata=None,
                    document=pd.Series(["20"]),
                    doc_id=pd.Series(["new_id"]),
                    doc_version=pd.Series([12]),
                )
            ]
        ]
    )

    assert store.max_version_for_doc("new_id") == 12

    clean_up_sql_lite_db(store)


@pytest.mark.parametrize("chunk_store", [SQLiteChunkStore, PandasChunkStore])
def test_get_doc_chunks(chunk_store):
    store = get_simple_chunk_store(chunk_store, use_metadata=False)

    store.insert(
        [
            [
                VersionedNewChunkBatch(
                    text=pd.Series(["a b c", "d e f", "g h i"]),
                    keywords=pd.Series(["a b c", "d e f", "g h i"]),
                    metadata=None,
                    document=pd.Series(["20", "21", "22"]),
                    doc_id=pd.Series(["new_id", "new_id", "new_id"]),
                    doc_version=pd.Series([10, 11, 12]),
                )
            ]
        ]
    )

    assert set([]) == set(store.get_doc_chunks(doc_id="new_id", before_version=10))
    assert set([7]) == set(store.get_doc_chunks(doc_id="new_id", before_version=11))
    assert set([7, 8]) == set(store.get_doc_chunks(doc_id="new_id", before_version=12))
    assert set([7, 8, 9]) == set(
        store.get_doc_chunks(doc_id="new_id", before_version=13)
    )
    assert set([7, 8, 9]) == set(
        store.get_doc_chunks(doc_id="new_id", before_version=float("inf"))
    )

    clean_up_sql_lite_db(store)
