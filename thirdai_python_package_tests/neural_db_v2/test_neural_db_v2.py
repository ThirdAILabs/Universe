import os
import shutil

import pandas as pd
import pytest
from ndbv2_utils import CSV_FILE, PDF_FILE, clean_up_sql_lite_db, load_chunks
from thirdai import neural_db_v2 as ndb
from thirdai.neural_db_v2.chunk_stores import PandasChunkStore, SQLiteChunkStore
from thirdai.neural_db_v2.retrievers import FinetunableRetriever, Mach, MachEnsemble

pytestmark = [pytest.mark.unit, pytest.mark.release]


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
