import os
import shutil

import pytest
from ndb_utils import train_simple_neural_db
from thirdai import neural_db

pytestmark = [pytest.mark.unit, pytest.mark.release]


def test_neural_db_save_load(train_simple_neural_db):
    ndb = train_simple_neural_db

    before_save_results = ndb.search(
        query="what color are apples",
        top_k=10,
    )

    if os.path.exists("temp"):
        shutil.rmtree("temp")

    ndb.save("temp")

    new_ndb = neural_db.NeuralDB.from_checkpoint("temp")

    after_save_results = new_ndb.search(
        query="what color are apples",
        top_k=10,
    )

    for r in after_save_results:
        print(r.text)
        print(r.score)

    for after, before in zip(after_save_results, before_save_results):
        assert after.text == before.text

    if os.path.exists("temp"):
        shutil.rmtree("temp")


def test_neural_db_reference_scores(train_simple_neural_db):
    ndb = train_simple_neural_db

    for r in ndb.search("what color are apples", top_k=10):
        assert 0 <= r.score and r.score <= 1
