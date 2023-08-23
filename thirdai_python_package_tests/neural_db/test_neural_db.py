import os
import shutil
from typing import List

import pytest
from ndb_utils import (
    create_simple_dataset,
    train_simple_neural_db,
    doc_choices,
    all_docs,
)
from thirdai import neural_db as ndb
from pathlib import Path

pytestmark = [pytest.mark.unit, pytest.mark.release]


def test_neural_db_reference_scores(train_simple_neural_db):
    db = train_simple_neural_db

    results = db.search("are apples green or red ?", top_k=10)
    for r in results:
        assert 0 <= r.score and r.score <= 1

    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def db_from_bazaar():
    bazaar = ndb.Bazaar(cache_dir=Path("."))
    bazaar.fetch()
    return bazaar.get_model("General QnA")


def get_upvote_target_id(db: ndb.NeuralDB, query: str, top_k: int):
    initial_ids = [r.id for r in db.search(query, top_k)]
    target_id = 0
    while target_id in initial_ids:
        target_id += 1
    return target_id


ARBITRARY_QUERY = "This is an arbitrary search query"


# Some of the following helper functions depend on others being called before them.
# It is best to call them in the order that these helper functions are written.
# They are only written as separate functions to make it easier to read.


def insert_works(db: ndb.NeuralDB, docs: List[ndb.Document]):
    db.insert(docs, train=False)
    assert len(db.sources()) == 4

    initial_scores = [r.score for r in db.search(ARBITRARY_QUERY, top_k=5)]

    db.insert(docs, train=True)
    assert len(db.sources()) == 4

    assert [r.score for r in db.search(ARBITRARY_QUERY, top_k=5)] != initial_scores


def search_works(db: ndb.NeuralDB, docs: List[ndb.Document], assert_acc: bool):
    top_k = 5
    correct_result = 0
    correct_source = 0
    for doc in docs:
        source = doc.reference(0).source
        for elem_id in range(doc.size):
            query = doc.reference(elem_id).text
            results = db.search(query, top_k)

            assert len(results) >= 1
            assert len(results) <= top_k

            for result in results:
                assert type(result.text) == str
                assert len(result.text) > 0

            correct_result += int(query in [r.text for r in results])
            correct_source += int(source in [r.source for r in results])

    assert correct_source / sum([doc.size for doc in docs]) > 0.8
    if assert_acc:
        assert correct_result / sum([doc.size for doc in docs]) > 0.8


def upvote_works(db: ndb.NeuralDB):
    # We have more than 10 indexed entities.
    target_id = get_upvote_target_id(db, ARBITRARY_QUERY, top_k=10)
    db.text_to_result(ARBITRARY_QUERY, target_id)
    assert target_id in [r.id for r in db.search(ARBITRARY_QUERY, top_k=10)]


def upvote_batch_works(db: ndb.NeuralDB):
    queries = [
        "This query is not related to any document.",
        "Neither is this one.",
        "Wanna get some biryani so we won't have to cook dinner?",
    ]
    target_ids = [get_upvote_target_id(db, query, top_k=10) for query in queries]
    db.text_to_result_batch(list(zip(queries, target_ids)))
    for query, target_id in zip(queries, target_ids):
        assert target_id in [r.id for r in db.search(query, top_k=10)]


def associate_works(db: ndb.NeuralDB):
    # Since this is still unstable, we only check that associate() updates the
    # model in *some* way, but we don't want to make stronger assertions as it
    # would make the test flaky.
    search_results = db.search(ARBITRARY_QUERY, top_k=5)
    initial_scores = [r.score for r in search_results]
    initial_ids = [r.id for r in search_results]

    another_arbitrary_query = "Eating makes me sleepy"
    db.associate(ARBITRARY_QUERY, another_arbitrary_query)

    new_search_results = db.search(ARBITRARY_QUERY, top_k=5)
    new_scores = [r.score for r in new_search_results]
    new_ids = [r.id for r in new_search_results]

    assert (initial_scores != new_scores) or (initial_ids != new_ids)


def save_load_works(db: ndb.NeuralDB):
    if os.path.exists("temp.ndb"):
        shutil.rmtree("temp.ndb")
    db.save("temp.ndb")
    search_results = [r.text for r in db.search(ARBITRARY_QUERY, top_k=5)]

    new_db = ndb.NeuralDB.from_checkpoint("temp.ndb")
    new_search_results = [r.text for r in new_db.search(ARBITRARY_QUERY, top_k=5)]

    assert search_results == new_search_results
    assert db.sources() == new_db.sources()

    shutil.rmtree("temp.ndb")


def clear_sources_works(db: ndb.NeuralDB):
    assert len(db.sources()) > 0
    db.clear_sources()
    assert len(db.sources()) == 0


def all_methods_work(db: ndb.NeuralDB, docs: List[ndb.Document], assert_acc: bool):
    insert_works(db, docs)
    search_works(db, docs, assert_acc)
    upvote_works(db)
    associate_works(db)
    save_load_works(db)
    clear_sources_works(db)


def test_neural_db_loads_from_model_bazaar():
    db_from_bazaar()


def test_neural_db_all_methods_work_on_new_model(all_docs):
    db = ndb.NeuralDB("user")
    all_methods_work(db, all_docs.values(), assert_acc=False)


def test_neural_db_all_methods_work_on_loaded_bazaar_model(all_docs):
    db = db_from_bazaar()
    all_methods_work(db, all_docs.values(), assert_acc=True)
