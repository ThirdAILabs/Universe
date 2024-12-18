import os
import random
import shutil
from typing import List
from uuid import uuid4

import pandas as pd
import pytest
import thirdai.neural_db_v2 as ndb
from ndbv2_utils import CSV_FILE, PDF_FILE
from test_finetunable_retriever_ndbv2 import (
    get_association_samples,
    get_supervised_samples,
    subsample_query,
)
from thirdai import search
from thirdai.neural_db_v2.supervised import InMemorySupervised

pytestmark = [pytest.mark.unit]


def dataset():
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../auto_ml/python_tests/texts.csv",
    )


@pytest.fixture(scope="function")
def fast_db(tmp_path):
    save_path = os.path.join(tmp_path, "tmp.db")
    db = ndb.FastDB(save_path=save_path)
    db.insert([ndb.CSV(path=dataset())])
    return db


def check_basic_query_accuracy(db: ndb.FastDB, dataset: pd.DataFrame):
    random.seed(64)
    for row in dataset.itertuples():
        query = subsample_query(row.text)
        assert db.search(query, top_k=1)[0][0].chunk_id == row.id


def compute_accuracy(db: ndb.FastDB, queries: List[str], labels: List[int]):
    correct = 0
    for query, label in zip(queries, labels):
        if db.search(query, top_k=1)[0][0].chunk_id == label:
            correct += 1
    return correct / len(labels)


@pytest.mark.release
def test_fast_db_search(fast_db):
    df = pd.read_csv(dataset())

    check_basic_query_accuracy(fast_db, df)

    for _, row in df.iterrows():
        id = row["id"]
        rank_results = fast_db.search(
            query=row["text"], constraints={"id": search.AnyOf([id, id + 1])}, top_k=1
        )
        assert id == rank_results[0][0].chunk_id


@pytest.mark.release
def test_fast_db_finetuning(fast_db):
    df = pd.read_csv(dataset())

    ids, acronyms = get_supervised_samples(df)

    acc_before_finetuning = compute_accuracy(fast_db, queries=acronyms, labels=ids)
    print("before finetuning: p@1 =", acc_before_finetuning)
    assert acc_before_finetuning <= 0.5

    fast_db.supervised_train(
        InMemorySupervised(queries=acronyms, ids=[[cid] for cid in ids])
    )

    acc_after_finetuning = compute_accuracy(fast_db, queries=acronyms, labels=ids)
    print("after finetuning: p@1 =", acc_after_finetuning)
    assert acc_after_finetuning >= 0.9


@pytest.mark.release
def test_fast_db_upvote(fast_db):
    df = pd.read_csv(dataset())

    ids, acronyms = get_supervised_samples(df)

    acc_before_upvote = compute_accuracy(fast_db, queries=acronyms, labels=ids)
    print("before upvote p@1 =", acc_before_upvote)
    assert acc_before_upvote <= 0.5

    fast_db.upvote(queries=acronyms, chunk_ids=ids)

    acc_after_upvote = compute_accuracy(fast_db, queries=acronyms, labels=ids)
    print("after upvote p@1 =", acc_after_upvote)
    assert acc_after_upvote >= 0.9


@pytest.mark.release
def test_fast_db_associate(fast_db):
    df = pd.read_csv(dataset())

    ids, acronyms, targets = get_association_samples(df)
    ids = ids.to_list()
    acronyms = acronyms.to_list()

    acc_before_associate = compute_accuracy(fast_db, queries=acronyms, labels=ids)
    print("before associate p@1 =", acc_before_associate)
    assert acc_before_associate <= 0.5

    fast_db.associate(sources=acronyms, targets=targets, associate_strength=1)

    acc_after_associate = compute_accuracy(fast_db, queries=acronyms, labels=ids)
    print("after associate p@1 =", acc_after_associate)
    assert acc_after_associate >= 0.9


@pytest.mark.release
def test_fast_db_save_load(tmp_path):
    db = ndb.FastDB(save_path=os.path.join(tmp_path, "tmp.db"))

    db.insert([ndb.CSV(CSV_FILE), ndb.PDF(PDF_FILE)])

    queries = ["lorem ipsum", "contrary"]
    results_before = db.search_batch(queries, top_k=10)

    save_file = os.path.join(tmp_path, "save.db")

    db.save(save_file)

    db = ndb.FastDB.load(save_file)

    results_after = db.search_batch(queries, top_k=10)

    assert results_before == results_after

    db.insert([ndb.CSV(CSV_FILE), ndb.PDF(PDF_FILE)])


@pytest.mark.release
def test_fast_db_doc_versioning(tmp_path):
    db = ndb.FastDB(save_path=os.path.join(tmp_path, "tmp.db"))

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

    db.delete_doc("b", keep_latest_version=True)
    check_results(db.search("v w x y z", top_k=5), "b", [3])

    db.delete_doc("a")
    check_results(db.search("a b c d e v", top_k=5), "b", [3])


# Test is not marked a release test because it causes mac wheel builds to timeout
def test_neural_db_v2_reranker(tmp_path):
    db = ndb.FastDB(save_path=os.path.join(tmp_path, "tmp.db"))

    db.insert([ndb.CSV(CSV_FILE)])

    regular = db.search("what are the roots of lorem ipsum", top_k=3)
    reranked = db.search("what are the roots of lorem ipsum", top_k=3, rerank=True)

    assert len(regular) > 0
    assert len(reranked) > 0
    assert len(regular) == len(reranked)
    assert [x[1] for x in regular] != [x[1] for x in reranked]


def test_neural_db_v2_with_splade(tmp_path):
    db = ndb.FastDB(save_path=os.path.join(tmp_path, "tmp.db"), splade=True)

    db.insert([ndb.CSV(CSV_FILE)])

    results = db.search("what are the roots of lorem ipsum", top_k=3)

    assert len(results) > 0
    assert results[0][0].chunk_id == 1
