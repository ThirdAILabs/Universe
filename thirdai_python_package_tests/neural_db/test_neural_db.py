import os
import shutil

import pytest
from ndb_utils import create_simple_dataset
from thirdai import neural_db

pytestmark = [pytest.mark.unit, pytest.mark.release]


def test_neural_db_save_load(create_simple_dataset):
    filename = create_simple_dataset
    ndb = neural_db.NeuralDB()

    doc = neural_db.CSV(
        filename,
        id_column="id",
        strong_columns=["text"],
        weak_columns=["text"],
        reference_columns=[],
    )

    ndb.insert(sources=[doc], train=True)

    before_save_results = ndb.search(
        query="some query",
        top_k=10,
    )

    if os.path.exists("temp"):
        shutil.rmtree("temp")

    ndb.save("temp")

    new_ndb = neural_db.NeuralDB.from_checkpoint("temp")

    after_save_results = new_ndb.search(
        query="some query",
        top_k=10,
    )

    for after, before in zip(after_save_results, before_save_results):
        assert after.text == before.text
