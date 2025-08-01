import os
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ndbv2_utils import CSV_FILE, PDF_FILE, clean_up_sql_lite_db, load_chunks
from thirdai import demos
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

    def generate_metadata_document(n_chunks: int, save_path: str, col_name_prefix: str):
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

        pd.DataFrame(
            {
                f"{col_name_prefix}_integer_col": integer_col,
                f"{col_name_prefix}_float_col": float_col,
                f"{col_name_prefix}_bool_col": bool_col,
                f"{col_name_prefix}_string_col": string_col,
                f"{col_name_prefix}_text_col": ["random text"] * n_chunks,
                f"{col_name_prefix}_keyword_col": ["random keyword"] * n_chunks,
            }
        ).to_csv(save_path, index=False)

    n_chunks = 100

    # create doc_a csv file
    doc_a_path = "doc_a.csv"
    generate_metadata_document(n_chunks, doc_a_path, col_name_prefix="a")

    doc_b_path = "doc_b.csv"
    generate_metadata_document(n_chunks, doc_b_path, col_name_prefix="b")

    db.insert(
        [
            ndb.CSV(
                doc_a_path,
                text_columns=["a_text_col"],
                keyword_columns=["a_keyword_col"],
                doc_id="a",
            ),
            ndb.CSV(
                doc_b_path,
                text_columns=["b_text_col"],
                keyword_columns=["b_keyword_col"],
                doc_id="b",
            ),
            # Don't insert any document with same metadata column. Reason: Comment on PandasChunkStore insert function last part.
        ]
    )

    summarized_metadata = db.chunk_store.document_metadata_summary.summarized_metadata

    for doc_id, csv_path in zip(["a", "b"], [doc_a_path, doc_b_path]):
        df = pd.read_csv(csv_path)
        for metadata_col_name in [f"{doc_id}_integer_col", f"{doc_id}_float_col"]:
            summary = summarized_metadata[(doc_id, 1)][metadata_col_name].summary
            assert df[metadata_col_name].min() == summary.min
            assert df[metadata_col_name].max() == summary.max

        for metadata_col_name in [f"{doc_id}_string_col", f"{doc_id}_bool_col"]:
            summary = summarized_metadata[(doc_id, 1)][metadata_col_name].summary
            assert summary.unique_values.issubset(set(df[metadata_col_name].unique()))

    # Also check that the loaded chunk_store have the same summarized_metadata
    save_file = "saved.ndb"
    db.save(save_file)

    loaded_db = ndb.NeuralDB.load(save_file)
    loaded_db_summarized_metadata = (
        loaded_db.chunk_store.document_metadata_summary.summarized_metadata
    )
    for doc_id, csv_path in zip(["a", "b"], [doc_a_path, doc_b_path]):
        df = pd.read_csv(csv_path)
        for metadata_col_name in [f"{doc_id}_integer_col", f"{doc_id}_float_col"]:
            summary = loaded_db_summarized_metadata[(doc_id, 1)][
                metadata_col_name
            ].summary
            assert df[metadata_col_name].min() == summary.min
            assert df[metadata_col_name].max() == summary.max

        for metadata_col_name in [f"{doc_id}_string_col", f"{doc_id}_bool_col"]:
            summary = loaded_db_summarized_metadata[(doc_id, 1)][
                metadata_col_name
            ].summary
            assert summary.unique_values.issubset(set(df[metadata_col_name].unique()))

    clean_up_sql_lite_db(db.chunk_store)
    shutil.rmtree(save_file)
    os.remove(doc_a_path)
    os.remove(doc_b_path)


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


def evaluate(
    db: ndb.NeuralDB,
    test_path: str,
    query_col: str = "QUERY",
    id_col: str = "DOC_ID",
    id_delim: str = ":",
):
    queries = pd.read_csv(test_path)
    queries[id_col] = queries[id_col].map(lambda x: list(map(int, x.split(id_delim))))

    correct = 0
    for _, row in queries.iterrows():
        results = db.search(row[query_col], top_k=2)

        if len(results) > 0 and results[0][0].metadata[id_col] in row[id_col]:
            correct += 1

    p_at_1 = correct / len(queries)
    print(f"p@1 = {p_at_1}")
    return p_at_1


@pytest.mark.release
def test_neural_db_v2_scifact_finetuning_explicit_validiation():
    docs, train, test, _ = demos.download_beir_dataset("scifact")

    db = ndb.NeuralDB()

    db.insert([ndb.CSV(docs, text_columns=["TITLE", "TEXT"])])

    acc_before = evaluate(db, test)
    assert 0.54 < acc_before < 0.56  # should be 0.553

    # This will make the accuracy after finetuning < 0.6
    db.retriever.retriever.set_lambda(0.2)

    db.supervised_train(
        ndb.supervised.CsvSupervised(
            train, query_column="QUERY", id_column="DOC_ID", id_delimiter=":"
        ),
        validation=ndb.supervised.CsvSupervised(
            test, query_column="QUERY", id_column="DOC_ID", id_delimiter=":"
        ),
    )

    # p@1 should be 0.767 (validation autotuning should fix lambda)
    acc_after = evaluate(db, test)
    assert 0.76 < acc_after < 0.78

    # This is to verify that the autotuned value is not 0.2
    db.retriever.retriever.set_lambda(0.2)
    assert evaluate(db, test) < 0.6  # Should be 0.583


@pytest.mark.release
def test_neural_db_v2_scifact_finetuning_automatic_validation_subsampling():
    docs, train, test, _ = demos.download_beir_dataset("scifact")

    db = ndb.NeuralDB()

    db.insert([ndb.CSV(docs, text_columns=["TITLE", "TEXT"])])

    acc_before = evaluate(db, test)
    assert 0.54 < acc_before < 0.56  # should be 0.553

    # This will make the accuracy after finetuning < 0.6
    db.retriever.retriever.set_lambda(0.2)

    # validation will be subsampled from train data
    db.supervised_train(
        ndb.supervised.CsvSupervised(
            train, query_column="QUERY", id_column="DOC_ID", id_delimiter=":"
        ),
    )

    np.random.seed(42)  # for validation subsampling
    # p@1 should be ~0.74-0.75 (validation split can affect accuracy a little)
    acc_after = evaluate(db, test)
    assert 0.7 < acc_after

    # This is to verify that the autotuned value is not 0.2
    db.retriever.retriever.set_lambda(0.2)
    assert evaluate(db, test) < 0.6  # Should be ~0.56-0.58
