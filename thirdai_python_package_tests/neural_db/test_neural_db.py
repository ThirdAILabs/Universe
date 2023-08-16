import os
import shutil

import pytest
from ndb_utils import create_simple_dataset, train_simple_neural_db
from thirdai import neural_db as ndb
from pathlib import Path

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

    new_ndb = ndb.NeuralDB.from_checkpoint("temp")

    after_save_results = new_ndb.search(
        query="what color are apples",
        top_k=10,
    )

    for after, before in zip(after_save_results, before_save_results):
        assert after.text == before.text
        assert after.score == before.score

    if os.path.exists("temp"):
        shutil.rmtree("temp")


def test_neural_db_reference_scores(train_simple_neural_db):
    ndb = train_simple_neural_db

    results = ndb.search("are apples green or red ?", top_k=10)
    for r in results:
        assert 0 <= r.score and r.score <= 1

    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def db_from_bazaar():
    bazaar = ndb.Bazaar(Path("."))
    bazaar.fetch()
    return bazaar.get_model("General QnA")


def get_upvote_target_id(db: ndb.NeuralDB, query: str):
    initial_ids = [r.id for r in db.search(query, top_k=10)]
    target_id = 0
    while target_id in initial_ids:
        target_id += 1
    return target_id


# Some of the following helper functions depend on others being called before them.
# It is best to call them in the order that these helper functions are written.
# They are only written as separate functions to make it easier to read.


def insert_works(db: ndb.NeuralDB):
    documents = [
        ndb.CSV(BLA),
        ndb.DOCX(BLA),
        ndb.PDF(BLA),
        ndb.URL(BLA),
    ]
    db.insert(documents, train=False)
    assert len(db.sources()) == 4

    initial_scores = [r.score for r in db.search(RANDOMSTRING, top_k=5)]

    db.insert(documents, train=True)
    assert len(db.sources()) == 4

    assert [r.score for r in db.search(RANDOMSTRING, top_k=5)] != initial_scores


def search_works(db: ndb.NeuralDB):
    # What edge cases can I test?
    # Empty string is not one since we don't support that
    # Maybe we should assert that it throws...
    # Maybe something from each of the sources (get a random reference)
    # then switch a percentage of characters with random characters
    # Assert that the real reference is in top k
    results = db.search(RANDOMSTRING, top_k=5)
    assert len(results) >= 1
    assert len(results) <= 5
    for result in results:
        assert type(result.text) == str
        assert len(result.text) > 0


def upvote_works(db: ndb.NeuralDB):
    # We have more than 10 indexed entities.
    target_id = get_upvote_target_id(db, RANDOMSTRING)
    db.text_to_result(RANDOMSTRING, target_id)
    assert target_id in [r.id for r in db.search(RANDOMSTRING, top_k=10)]


def upvote_batch_works(db: ndb.NeuralDB):
    RANDOMSTRINGS = []
    target_ids = [get_upvote_target_id(db, query) for query in RANDOMSTRINGS]
    db.text_to_result_batch(list(zip(RANDOMSTRINGS, target_ids)))
    for query, target_id in zip(RANDOMSTRINGS, target_ids):
        assert target_id in [r.id for r in db.search(query, top_k=10)]


def associate_works(db: ndb.NeuralDB):
    # Since this is still unstable, we only check that associate() updates the
    # model in *some* way, but we don't want to make stronger assertions as it
    # would make the test flaky.
    search_results = db.search(RANDOMSTRING, top_k=5)
    initial_scores = [r.score for r in search_results]
    initial_ids = [r.id for r in search_results]

    db.associate(RANDOMSTRING, RANDOMSTRING2)

    new_search_results = db.search(RANDOMSTRING, top_k=5)
    new_scores = [r.score for r in new_search_results]
    new_ids = [r.id for r in new_search_results]

    assert (initial_scores != new_scores) or (initial_ids != new_ids)


def save_load_works(db: ndb.NeuralDB):
    db.save("temp.ndb")
    search_results = db.search(RANDOMSTRING, top_k=5)

    new_db = ndb.NeuralDB.from_checkpoint("temp.ndb")
    new_search_results = new_db.search(RANDOMSTRING, top_k=5)

    assert search_results == new_search_results
    assert db.sources() == new_db.sources()


def clear_sources_works(db: ndb.NeuralDB):
    assert len(db.sources()) > 0
    db.clear_sources()
    assert len(db.sources()) == 0


def all_methods_work(db: ndb.NeuralDB):
    insert_works(db)
    search_works(db)
    upvote_works(db)
    associate_works(db)
    save_load_works(db)
    clear_sources_works(db)


def test_neural_db_loads_from_model_bazaar():
    db_from_bazaar()


def test_neural_db_all_methods_work_on_new_model():
    db = ndb.NeuralDB("user")
    all_methods_work(db)


def test_neural_db_insert_works_on_loaded_bazaar_model():
    db = db_from_bazaar()
    all_methods_work(db)
