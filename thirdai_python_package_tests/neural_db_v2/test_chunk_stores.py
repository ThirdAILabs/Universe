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
    InRange,
    LessThan,
    NoneOf,
    Substring,
)
from thirdai.neural_db_v2.core.types import NewChunkBatch
from thirdai.neural_db_v2.documents import InMemoryText, PrebatchedDoc

pytestmark = [pytest.mark.release]


def assert_chunk_ids(inserted_batches, expected_chunk_ids):
    actual_chunk_ids = []
    for batch in inserted_batches:
        for chunk_id in batch.chunk_id:
            actual_chunk_ids.append(chunk_id)

    assert expected_chunk_ids == actual_chunk_ids


def get_simple_chunk_store(chunk_store_type, use_metadata=True):
    store = chunk_store_type()

    docs = [
        PrebatchedDoc(
            [
                NewChunkBatch(
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
                ),
                NewChunkBatch(
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
                ),
            ]
        ),
        PrebatchedDoc(
            [
                NewChunkBatch(
                    text=pd.Series(["5 6", "7 8"]),
                    keywords=pd.Series(["55 66", "77 88"]),
                    document=pd.Series(["doc5", "doc6"]),
                    metadata=(
                        pd.DataFrame(
                            {
                                "class": ["x", "y"],
                                "time": [6.5, 2.4],
                                "item": ["t", "u"],
                            }
                        )
                        if use_metadata
                        else None
                    ),
                ),
            ]
        ),
    ]

    inserted_batches, metadata = store.insert(docs)

    assert len(docs) == len(metadata)

    for doc, doc_metadata in zip(docs, metadata):
        index = 0
        assert doc_metadata.doc_id == doc.doc_id()
        for batch in doc.chunks():
            for i in range(len(batch)):
                chunk = store.get_chunks([doc_metadata.chunk_ids[index]])[0]

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
        metadata={"class": "b", "number": 9, "item": "y"},
    )
    check_chunk_contents(
        chunks[1],
        chunk_id=3,
        value=3,
        metadata={"class": "b", "number": 2, "time": 2.6},
    )
    check_chunk_contents(
        chunks[2],
        chunk_id=0,
        value=0,
        metadata={"class": "a", "number": 4, "item": "x"},
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
        metadata={"class": "b", "number": 2, "time": 2.6},
    )
    check_chunk_contents(
        chunks[1],
        chunk_id=0,
        value=0,
        metadata={"class": "a", "number": 4, "item": "x"},
    )

    new_batches = PrebatchedDoc(
        [
            NewChunkBatch(
                text=pd.Series(["8 9", "1 2"]),
                keywords=pd.Series(["88 99", "11 22"]),
                document=pd.Series(["doc8", "doc1"]),
                metadata=pd.DataFrame(
                    {"class": ["c", "d"], "time": [7.2, 8.1], "item": ["y", "z"]}
                ),
            ),
        ]
    )

    inserted_batches, _ = store.insert([new_batches])
    assert_chunk_ids(inserted_batches, [7, 8])

    chunks = store.get_chunks([8, 0, 7])
    check_chunk_contents(
        chunks[0],
        chunk_id=8,
        value=1,
        metadata={"class": "d", "time": 8.1, "item": "z"},
    )
    check_chunk_contents(
        chunks[1],
        chunk_id=0,
        value=0,
        metadata={"class": "a", "number": 4, "item": "x"},
    )
    check_chunk_contents(
        chunks[2],
        chunk_id=7,
        value=8,
        metadata={"class": "c", "time": 7.2, "item": "y"},
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
        metadata={"class": "d", "time": 8.1, "item": "z"},
    )

    os.remove(path)


@pytest.mark.parametrize("chunk_store", [SQLiteChunkStore, PandasChunkStore])
def test_chunk_store_missing_columns(chunk_store):
    store = get_simple_chunk_store(chunk_store)

    with pytest.raises(KeyError, match="Missing columns in metadata: BADCOLUMN"):
        store.filter_chunk_ids({"BADCOLUMN": EqualTo("b")})

    clean_up_sql_lite_db(store)


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
def test_chunk_store_constraints_none_of(chunk_store):
    store = get_simple_chunk_store(chunk_store)

    chunk_ids = store.filter_chunk_ids({"number": NoneOf([4, 9])})
    assert chunk_ids == set([2, 3])

    clean_up_sql_lite_db(store)


@pytest.mark.parametrize("chunk_store", [SQLiteChunkStore, PandasChunkStore])
def test_chunk_store_constraint_in_range(chunk_store):
    store = get_simple_chunk_store(chunk_store)

    chunk_ids = store.filter_chunk_ids(
        {"number": InRange(min_value=2, max_value=4, min_inclusive=False)}
    )
    assert chunk_ids == set([0, 4])

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

    new_batch = NewChunkBatch(
        text=pd.Series(["0", "1", "2"]),
        keywords=pd.Series(["00 11", "11 22", "22 33"]),
        document=pd.Series(["doc0", "doc1", "doc2"]),
        metadata=pd.DataFrame(
            {"class": ["a", "b", "c"], "number": [4, 9, 11], "item": ["x", "y", "z"]}
        ),
    )

    inserted_batches_1, _ = store.insert([PrebatchedDoc([new_batch])])
    inserted_batches_2, _ = store.insert([PrebatchedDoc([new_batch])])

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

    doc = PrebatchedDoc(
        [
            NewChunkBatch(
                text=pd.Series(["a b c"]),
                keywords=pd.Series(["a b c"]),
                metadata=None,
                document=pd.Series(["20"]),
            )
        ],
        doc_id="new_id",
    )
    store.insert([doc])
    assert store.max_version_for_doc("new_id") == 1

    store.insert([doc])
    assert store.max_version_for_doc("new_id") == 2

    clean_up_sql_lite_db(store)


@pytest.mark.parametrize("chunk_store", [SQLiteChunkStore, PandasChunkStore])
def test_get_doc_chunks(chunk_store):
    store = get_simple_chunk_store(chunk_store, use_metadata=False)

    doc = PrebatchedDoc(
        [
            NewChunkBatch(
                text=pd.Series(["a b c"]),
                keywords=pd.Series(["a b c"]),
                metadata=None,
                document=pd.Series(["20"]),
            )
        ],
        doc_id="new_id",
    )
    store.insert([doc])
    store.insert([doc])
    store.insert([doc])

    assert set([]) == set(store.get_doc_chunks(doc_id="new_id", before_version=1))
    assert set([7]) == set(store.get_doc_chunks(doc_id="new_id", before_version=2))
    assert set([7, 8]) == set(store.get_doc_chunks(doc_id="new_id", before_version=3))
    assert set([7, 8, 9]) == set(
        store.get_doc_chunks(doc_id="new_id", before_version=4)
    )
    assert set([7, 8, 9]) == set(
        store.get_doc_chunks(doc_id="new_id", before_version=float("inf"))
    )

    clean_up_sql_lite_db(store)


def test_multivalue_metadata():
    store = SQLiteChunkStore()

    docs = [
        InMemoryText(
            document_name="1",
            text=["a b c", "d e f"],
            chunk_metadata=[
                {"start": "a", "items": [1, 3]},
                {"end": "f", "items": [2, 4]},
            ],
            doc_metadata={"time": 20, "permissions": ["group1", "group2", "group3"]},
        ),
        InMemoryText(
            document_name="2",
            text=["r s t", "u v w", "x y z"],
            chunk_metadata=[
                {"start": "r", "end": "t", "items": [2, 3]},
                {"start": "u", "end": "w"},
                {"start": "x", "items": [1, 5]},
            ],
            doc_metadata={"type": "pdf", "permissions": ["group1", "group4"]},
        ),
    ]
    store.insert(docs)

    chunk_ids = store.filter_chunk_ids(
        {
            "permissions": AnyOf(["group2", "group10"]),
        }
    )
    assert chunk_ids == set([0, 1])

    chunk_ids = store.filter_chunk_ids(
        {
            "permissions": AnyOf(["group2", "group4"]),
        }
    )
    assert chunk_ids == set([0, 1, 2, 3, 4])

    chunk_ids = store.filter_chunk_ids(
        {
            "items": AnyOf([2, 5]),
        }
    )
    assert chunk_ids == set([1, 2, 4])

    chunk_ids = store.filter_chunk_ids(
        {
            "permissions": EqualTo("group1"),
            "end": AnyOf(["w", "t"]),
        }
    )
    assert chunk_ids == set([2, 3])

    chunk_ids = store.filter_chunk_ids(
        {
            "permissions": EqualTo("group3"),
            "items": GreaterThan(4),
        }
    )
    assert chunk_ids == set([1])

    chunk_ids = store.filter_chunk_ids(
        {
            "items": GreaterThan(3),
            "permissions": EqualTo("group4"),
        }
    )
    assert chunk_ids == set([2, 4])

    chunk_ids = store.filter_chunk_ids(
        {
            "items": GreaterThan(3),
            "permissions": Substring("oup4"),
        }
    )
    assert chunk_ids == set([2, 4])

    chunk_ids = store.filter_chunk_ids(
        {
            "items": GreaterThan(4),
            "time": LessThan(25),
        }
    )
    assert chunk_ids == set([1])

    chunk_ids = store.filter_chunk_ids(
        {
            "permissions": EqualTo("group4"),
            "start": AnyOf(["r", "x"]),
            "items": GreaterThan(4),
        }
    )
    assert chunk_ids == set([4])

    chunk_ids = store.filter_chunk_ids(
        {
            "permissions": EqualTo("group4"),
            "start": AnyOf(["r", "x"]),
            "items": AnyOf([2, 3]),
        }
    )
    assert chunk_ids == set([2])

    chunk_ids = store.filter_chunk_ids(
        {
            "permissions": Substring("oup4"),
            "start": AnyOf(["r", "x"]),
            "items": AnyOf([2, 3]),
        }
    )
    assert chunk_ids == set([2])

    chunks = store.get_chunks([0, 4, 3])

    chunk_0_metadata = {
        "start": "a",
        "items": [1, 3],
        "time": 20,
        "permissions": ["group1", "group2", "group3"],
    }
    assert chunks[0].metadata == chunk_0_metadata

    chunk_4_metadata = {
        "start": "x",
        "items": [1, 5],
        "type": "pdf",
        "permissions": ["group1", "group4"],
    }
    assert chunks[1].metadata == chunk_4_metadata

    chunk_3_metadata = {
        "start": "u",
        "end": "w",
        "type": "pdf",
        "permissions": ["group1", "group4"],
    }
    assert chunks[2].metadata == chunk_3_metadata


def test_encryption():
    key = "209402"
    store = SQLiteChunkStore(encryption_key=key)

    texts = [
        "apples are an excellent fruit",
        "i like nectarines",
        "mangos are tasty too",
    ]
    doc = PrebatchedDoc(
        [
            NewChunkBatch(
                text=pd.Series(texts),
                keywords=pd.Series(["apples", "nectarines", "mangos"]),
                metadata=None,
                document=pd.Series(["a", "b", "c"]),
            )
        ],
    )

    store.insert([doc])

    def check_queries(chunk_store, check_equal):
        chunks = chunk_store.get_chunks([0, 1, 2])

        for chunk, text in zip(chunks, texts):
            assert (chunk.text == text) == check_equal

    check_queries(store, check_equal=True)

    save_path = store.db_name + ".tmp.save"
    store.save(save_path)

    store_wo_key = SQLiteChunkStore.load(save_path)
    check_queries(store_wo_key, check_equal=False)

    with pytest.raises(ValueError, match="Invalid decryption key"):
        store_w_wrong_key = SQLiteChunkStore.load(save_path, encryption_key=key + "0")

    store_w_key = SQLiteChunkStore.load(save_path, encryption_key=key)
    check_queries(store_w_key, check_equal=True)

    os.remove(save_path)


@pytest.mark.parametrize("chunk_store", [SQLiteChunkStore, PandasChunkStore])
def test_list_documents(chunk_store):
    store = chunk_store()

    store.insert(
        [
            InMemoryText(document_name="a.txt", text=["a", "aa"], doc_id="a1"),
            InMemoryText(document_name="b.txt", text=["b", "bb"], doc_id="b1"),
            InMemoryText(document_name="c.txt", text=["c", "cc"], doc_id="c1"),
            InMemoryText(document_name="b.txt", text=["b", "bb"], doc_id="b2"),
            InMemoryText(document_name="a.txt", text=["a", "aa"], doc_id="a1"),
            InMemoryText(document_name="c.txt", text=["c", "cc"], doc_id="c1"),
        ]
    )

    expected_docs = [
        {"doc_id": "a1", "doc_version": 1, "document": "a.txt"},
        {"doc_id": "a1", "doc_version": 2, "document": "a.txt"},
        {"doc_id": "b1", "doc_version": 1, "document": "b.txt"},
        {"doc_id": "b2", "doc_version": 1, "document": "b.txt"},
        {"doc_id": "c1", "doc_version": 1, "document": "c.txt"},
        {"doc_id": "c1", "doc_version": 2, "document": "c.txt"},
    ]

    def map_to_tuples(sources):
        return [(x["doc_id"], x["doc_version"], x["document"]) for x in sources]

    docs = map_to_tuples(store.documents())
    expected_docs = map_to_tuples(expected_docs)

    assert len(docs) == len(expected_docs)
    assert set(docs) == set(expected_docs)


@pytest.mark.parametrize("chunk_store", [SQLiteChunkStore, PandasChunkStore])
def test_chunk_context(chunk_store):
    store = chunk_store()

    store.insert(
        [
            InMemoryText(document_name="b.txt", text=["b1", "b2", "b3"], doc_id="b1"),
            InMemoryText(
                document_name="a.txt", text=["a1", "a2", "a3", "a4"], doc_id="a1"
            ),
            InMemoryText(document_name="a.txt", text=["a5", "a6"], doc_id="a1"),
        ]
    )

    b1, a1, a3, a5 = store.get_chunks([0, 3, 5, 7])

    def get_context_text(chunk, radius):
        context = store.context(chunk, radius)
        return [c.text for c in context]

    assert get_context_text(b1, 2) == ["b1", "b2", "b3"]

    # Context should not go beyond document (preceding or subsequent doc)
    assert get_context_text(b1, 4) == ["b1", "b2", "b3"]
    assert get_context_text(a1, 2) == ["a1", "a2", "a3"]

    # Check context in the middle
    assert get_context_text(a3, 1) == ["a2", "a3", "a4"]

    # Context should not go to different version of same doc
    assert get_context_text(a3, 2) == ["a1", "a2", "a3", "a4"]
    assert get_context_text(a5, 1) == ["a5", "a6"]
