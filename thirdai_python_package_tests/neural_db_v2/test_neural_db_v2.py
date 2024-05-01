import os
import shutil

import pytest
from ndbv2_utils import CSV_FILE, PDF_FILE, load_chunks
from thirdai import neural_db_v2 as ndb
from thirdai.neural_db_v2.chunk_stores import PandasChunkStore, SQLiteChunkStore
from thirdai.neural_db_v2.retrievers import FinetunableRetriever, Mach

pytestmark = [pytest.mark.unit, pytest.mark.release]


@pytest.mark.parametrize(
    "chunk_store, retriever",
    [
        (SQLiteChunkStore, FinetunableRetriever),
        (PandasChunkStore, Mach),
    ],
)
def test_neural_db_v2_save_load_integration(chunk_store, retriever):
    db = ndb.NeuralDB(chunk_store=chunk_store(), retriever=retriever())

    db.insert([ndb.CSV(CSV_FILE), ndb.PDF(PDF_FILE)])

    queries = ["lorem ipsum", "contrary"]
    results_before = db.search(queries, top_k=10)

    save_file = "neural_db_v2_save_file.ndb"

    if os.path.exists(save_file):
        shutil.rmtree(save_file)

    db.save(save_file)
    db = ndb.NeuralDB.load(save_file)

    results_after = db.search(queries, top_k=10)

    assert results_before == results_after

    db.insert([ndb.CSV(CSV_FILE), ndb.PDF(PDF_FILE)])

    shutil.rmtree(save_file)


@pytest.mark.parametrize(
    "chunk_store, retriever",
    [
        # (SQLiteChunkStore, FinetunableRetriever),
        (PandasChunkStore, Mach),
    ],
)
def test_neural_db_v2_supervised_training(chunk_store, retriever, load_chunks):
    db = ndb.NeuralDB(chunk_store=chunk_store(), retriever=retriever())

    db.insert(
        [
            ndb.InMemoryText(
                document_name="texts",
                text=load_chunks["text"],
                custom_id=[f"custom{i}" for i in range(len(load_chunks))],
                doc_metadata=[{"a": "b"} for i in range(len(load_chunks))],
            )
        ]
    )

    queries = ["alpha beta phi", "epsilon omega phi delta"]

    for uses_db_id in [True, False]:
        ids = [[1], [2]] if uses_db_id else [["custom1"], ["custom2"]]

        db.supervised_train(
            ndb.InMemorySupervised(queries=queries, ids=ids, uses_db_id=uses_db_id)
        )
