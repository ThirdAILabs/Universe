import os
import random
import shutil
from pathlib import Path

import pandas as pd
import pytest
from ndbv2_utils import CSV_FILE, PDF_FILE, clean_up_sql_lite_db, load_chunks
from thirdai import neural_db_v2 as ndb
from thirdai.neural_db_v2.chunk_stores import PandasChunkStore, SQLiteChunkStore
from thirdai.neural_db_v2.retrievers import FinetunableRetriever, Mach, MachEnsemble

pytestmark = [pytest.mark.unit]


@pytest.mark.release
@pytest.mark.parametrize(
    "chunk_store, retriever",
    [
        (SQLiteChunkStore, FinetunableRetriever),
        (PandasChunkStore, FinetunableRetriever),
        (PandasChunkStore, Mach),
        (PandasChunkStore, lambda: MachEnsemble(n_models=2, n_buckets=10_000)),
    ],
)
def test_neural_db_v2_save_load_integration(chunk_store, retriever):
    db = ndb.NeuralDB(chunk_store=chunk_store(), retriever=retriever())

    db.insert([ndb.CSV(CSV_FILE), ndb.PDF(PDF_FILE)], epochs=4)

    queries = ["lorem ipsum", "contrary"]
    results_before = db.search_batch(queries, top_k=10)

    save_file = "neural_db_v2_save_file.ndb"

    if os.path.exists(save_file):
        shutil.rmtree(save_file)

    db.save(save_file)

    if isinstance(db.chunk_store, SQLiteChunkStore):
        os.remove(os.path.basename(db.chunk_store.db_name))

    db = ndb.NeuralDB.load(save_file)

    results_after = db.search_batch(queries, top_k=10)

    assert results_before == results_after

    db.insert([ndb.CSV(CSV_FILE), ndb.PDF(PDF_FILE)], epochs=1)

    shutil.rmtree(save_file)


@pytest.mark.release
@pytest.mark.parametrize(
    "chunk_store, retriever",
    [
        (SQLiteChunkStore, FinetunableRetriever),
        (PandasChunkStore, Mach),
        (PandasChunkStore, lambda: MachEnsemble(n_models=2, n_buckets=10_000)),
    ],
)
def test_neural_db_v2_supervised_training(chunk_store, retriever, load_chunks):
    db = ndb.NeuralDB(chunk_store=chunk_store(), retriever=retriever())

    db.insert(
        [
            ndb.InMemoryText(
                document_name="texts",
                text=load_chunks["text"],
                chunk_metadata=[{"a": "b"} for i in range(len(load_chunks))],
            )
        ]
    )

    queries = ["alpha beta phi", "epsilon omega phi delta"]

    ids = [[1], [2, 3]]

    db.supervised_train(ndb.InMemorySupervised(queries=queries, ids=ids))

    ids = ["1", "2:3"]
    df = pd.DataFrame({"queries": queries, "ids": ids})
    csv_file_name = "test_neural_db_v2_supervised_training.csv"
    df.to_csv(csv_file_name, index=False)

    db.supervised_train(
        ndb.CsvSupervised(
            path=csv_file_name,
            query_column="queries",
            id_column="ids",
            id_delimiter=":",
        )
    )

    os.remove(csv_file_name)

    clean_up_sql_lite_db(db.chunk_store)


@pytest.mark.release
@pytest.mark.parametrize("chunk_store", [SQLiteChunkStore, PandasChunkStore])
def test_summarized_metadata(chunk_store):
    db = ndb.NeuralDB(chunk_store=chunk_store(), retriever=FinetunableRetriever())

    def generate_chunk_metadata(n_chunks: int):
        # prepare integer col
        integer_col = [random.randint(-200, 200) for _ in range(n_chunks)]

        # prepare bool col
        bool_col = random.choice([[True, False], [True], [False]])
        bool_col = bool_col * (n_chunks // len(bool_col))

        # prepare float col
        float_col = [random.uniform(-200, 200) for _ in range(n_chunks)]

        # prepare string col
        values = [
            "apple",
            "banana",
            "cherry",
            "date",
            "elderberry",
            "fig",
            "grape",
            "honeydew",
            "kiwi",
            "lemon",
        ]
        string_col = random.sample(values, k=min(n_chunks, 5))

        if len(string_col) < n_chunks:
            string_col = string_col * ((n_chunks // len(string_col)) + 1)
            string_col = random.sample(string_col, k=n_chunks)

        return integer_col, float_col, bool_col, string_col

    n_chunks = 100
    doc_a_metadata = generate_chunk_metadata(n_chunks)
    doc_b_metadata = generate_chunk_metadata(n_chunks)

    doc_metadata = {
        "a": {
            "doc_a_integer": doc_a_metadata[0],
            "doc_a_float": doc_a_metadata[1],
            "doc_a_bool": doc_a_metadata[2],
            "doc_a_string": doc_a_metadata[3],
        },
        "b": {
            "doc_b_integer": doc_b_metadata[0],
            "doc_b_float": doc_b_metadata[1],
            "doc_b_bool": doc_b_metadata[2],
            "doc_b_string": doc_b_metadata[3],
        },
    }
    db.insert(
        [
            ndb.InMemoryText(
                "doc_a",
                text=["Random text"] * n_chunks,
                chunk_metadata=doc_metadata["a"],
                doc_id="a",
            ),
            ndb.InMemoryText(
                "doc_b",
                text=["Another random text"] * n_chunks,
                chunk_metadata=doc_metadata["b"],
                doc_id="b",
            ),
        ]
    )

    summarized_metadata = db.chunk_store.summarized_metadata

    for doc_id in ["a", "b"]:
        for metadata_col_name, metadata_values in doc_metadata[doc_id].items():
            summary = summarized_metadata[doc_id][1][metadata_col_name].summary
            if metadata_col_name in ["doc_a_integer", "doc_a_float"]:
                assert min(metadata_values) == summary.min
                assert max(metadata_values) == summary.max
            else:
                assert summary.unique_values == set(metadata_values)

    # Also check that the loaded chunk_store have the same summarized_metadata
    db.save("saved.ndb")

    loaded_db = db.load("saved.ndb")
    loaded_db_summarized_metadata = loaded_db.chunk_store.summarized_metadata
    for doc_id in ["a", "b"]:
        for metadata_col_name, metadata_values in doc_metadata[doc_id].items():
            summary = loaded_db_summarized_metadata[doc_id][1][
                metadata_col_name
            ].summary
            if metadata_col_name in ["doc_a_integer", "doc_a_float"]:
                assert min(metadata_values) == summary.min
                assert max(metadata_values) == summary.max
            else:
                assert summary.unique_values == set(metadata_values)

    clean_up_sql_lite_db(db.chunk_store)
    clean_up_sql_lite_db(loaded_db.chunk_store)


@pytest.mark.release
@pytest.mark.parametrize("chunk_store", [SQLiteChunkStore, PandasChunkStore])
def test_neural_db_v2_doc_versioning(chunk_store):
    db = ndb.NeuralDB(chunk_store=chunk_store(), retriever=FinetunableRetriever())

    db.insert(
        [
            ndb.InMemoryText("doc_a", text=["a b c d e"], doc_id="a"),
            ndb.InMemoryText("doc_a", text=["a b c d"], doc_id="a"),
            ndb.InMemoryText("doc_b", text=["v w x y z"], doc_id="b"),
            ndb.InMemoryText("doc_a", text=["a b c"], doc_id="a"),
        ]
    )

    def check_results(results, doc_id, versions):
        assert len(results) == len(versions)
        for ver, res in zip(versions, results):
            assert res[0].doc_id == doc_id
            assert res[0].doc_version == ver

    check_results(db.search("v w x y z", top_k=5), "b", [1])

    db.insert(
        [
            ndb.InMemoryText("doc_b", text=["v w x y"], doc_id="b"),
            ndb.InMemoryText("doc_b", text=["v w x"], doc_id="b"),
        ]
    )

    check_results(db.search("a b c d e", top_k=5), "a", [1, 2, 3])
    check_results(db.search("v w x y z", top_k=5), "b", [1, 2, 3])

    deleted_chunks = db.delete_doc(
        "b", keep_latest_version=True, return_deleted_chunks=True
    )
    assert len(deleted_chunks) == 2
    assert set([c.chunk_id for c in deleted_chunks]) == set([2, 4])
    check_results(db.search("v w x y z", top_k=5), "b", [3])

    db.delete_doc("a")
    check_results(db.search("a b c d e v", top_k=5), "b", [3])

    clean_up_sql_lite_db(db.chunk_store)


@pytest.mark.release
def test_neural_db_v2_on_disk():
    save_path = "test_neural_db_v2_on_disk"

    db = ndb.NeuralDB(save_path=save_path)

    save_path = Path(save_path)
    assert os.path.exists(save_path / "chunk_store")
    assert os.path.exists(save_path / "retriever")
    assert os.path.exists(save_path / "metadata.json")

    del db

    db = ndb.NeuralDB.load(str(save_path))

    db.insert([ndb.CSV(CSV_FILE), ndb.PDF(PDF_FILE)])

    queries = ["lorem ipsum", "contrary"]
    results_before = db.search_batch(queries, top_k=10)

    del db

    db = ndb.NeuralDB.load(str(save_path))

    queries = ["lorem ipsum", "contrary"]
    results_after = db.search_batch(queries, top_k=10)

    assert results_before == results_after

    shutil.rmtree(save_path)


# Test is not marked a release test because it causes mac wheel builds to timeout
def test_neural_db_v2_reranker():
    db = ndb.NeuralDB()

    db.insert([ndb.CSV(CSV_FILE)])

    regular = db.search("what are the roots of lorem ipsum", top_k=3)
    reranked = db.search("what are the roots of lorem ipsum", top_k=3, rerank=True)

    assert len(regular) > 0
    assert len(reranked) > 0
    assert len(regular) == len(reranked)
    assert [x[1] for x in regular] != [x[1] for x in reranked]


def test_neural_db_v2_with_splade():
    db = ndb.NeuralDB(splade=True)

    db.insert([ndb.CSV(CSV_FILE)])

    results = db.search("what are the roots of lorem ipsum", top_k=3)

    assert len(results) > 0
    assert results[0][0].chunk_id == 1
