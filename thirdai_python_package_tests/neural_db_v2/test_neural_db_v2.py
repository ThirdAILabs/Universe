import pytest
from ndbv2_utils import CSV_FILE, PDF_FILE
from thirdai import neural_db_v2 as ndb
from thirdai.neural_db_v2.chunk_stores import PandasChunkStore, SQLiteChunkStore
from thirdai.neural_db_v2.retrievers import FinetunableRetriever, Mach

pytestmark = [pytest.mark.unit, pytest.mark.release]


@pytest.mark.parametrize(
    "chunk_store, retriever",
    [
        (SQLiteChunkStore, Mach),
        (SQLiteChunkStore, FinetunableRetriever),
        (PandasChunkStore, Mach),
        (PandasChunkStore, FinetunableRetriever),
    ],
)
def test_neural_db_v2_save_load_integration(chunk_store, retriever):
    db = ndb.NeuralDB(chunk_store=chunk_store(), retreiver=retriever())

    db.insert([ndb.CSV(CSV_FILE), ndb.PDF(PDF_FILE)])

    results_before = db.search(["lorem ipsum"], top_k=10)

    save_file = "neural_db_v2_save_file.ndb"
    db.save(save_file)
    db = ndb.NeuralDB.load(save_file)

    results_after = db.search(["lorem ipsum"], top_k=10)

    assert results_before == results_after
