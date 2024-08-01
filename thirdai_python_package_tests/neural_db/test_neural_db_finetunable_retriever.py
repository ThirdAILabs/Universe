import os
import random
import shutil

import pandas as pd
import pytest
from thirdai import neural_db as ndb

pytestmark = [pytest.mark.unit, pytest.mark.release]


def textfile():
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../auto_ml/python_tests/texts.csv",
    )


def build_db(on_disk):
    db = ndb.NeuralDB(on_disk=on_disk)

    db.insert([ndb.CSV(path=textfile(), id_column="id", weak_columns=["text"])])

    return db


def subsample_query(text, k=20):
    return " ".join(random.choices(text.split(), k=k))


def check_basic_query_accuracy(db):
    random.seed(64)
    for row in pd.read_csv(textfile()).itertuples():
        query = subsample_query(row.text)
        assert db.search(query, top_k=1)[0].id == row.id


@pytest.mark.parametrize("on_disk", [True, False])
def test_ndb_finetunable_retriever_search(on_disk):
    db = build_db(on_disk)
    check_basic_query_accuracy(db)

    for _, row in pd.read_csv(textfile()).iterrows():
        id = row["id"]
        rank_results = db._savable_state.model.score(
            [row["text"]],
            entities=[[id + 2, id, id + 1]],
            n_results=1,
        )[0]
        assert id == rank_results[0][0]


def compute_accuracy(db, queries, labels):
    correct = 0
    for query, label in zip(queries, labels):
        if db.search(query, top_k=1)[0].id == label:
            correct += 1
    return correct / len(labels)


def get_supervised_samples():
    df = pd.read_csv(textfile())

    random_words = []
    for sample in df["text"]:
        words = sample.split()
        random_words.extend(random.choices(words, k=4))

    acronyms = (
        df["text"]
        .map(
            lambda s: "".join(w[0] for w in s.split())
            + " "
            + " ".join(random.choices(random_words, k=5))
        )
        .to_list()
    )
    ids = df["id"].to_list()

    return ids, acronyms


@pytest.mark.parametrize("on_disk", [True, False])
def test_ndb_finetunable_retriever_finetuning(on_disk):
    db = build_db(on_disk)

    ids, acronyms = get_supervised_samples()

    acc_before_finetuning = compute_accuracy(db, queries=acronyms, labels=ids)
    print("before finetuning: p@1 =", acc_before_finetuning)
    assert acc_before_finetuning <= 0.5

    db.supervised_train_with_ref_ids(queries=acronyms, labels=[[y] for y in ids])

    acc_after_finetuning = compute_accuracy(db, queries=acronyms, labels=ids)
    print("after finetuning: p@1 =", acc_after_finetuning)
    assert acc_after_finetuning >= 0.9


@pytest.mark.parametrize("on_disk", [True, False])
def test_ndb_finetunable_retriever_upvote(on_disk):
    db = build_db(on_disk)

    ids, acronyms = get_supervised_samples()

    acc_before_upvote = compute_accuracy(db, queries=acronyms, labels=ids)
    print("before upvote p@1 =", acc_before_upvote)
    assert acc_before_upvote <= 0.5

    db.text_to_result_batch(list(zip(acronyms, ids)))

    acc_after_upvote = compute_accuracy(db, queries=acronyms, labels=ids)
    print("after upvote p@1 =", acc_after_upvote)
    assert acc_after_upvote >= 0.9


def get_association_samples():
    df = pd.read_csv(textfile())

    ids, acronyms = get_supervised_samples()

    targets = (
        df["text"].map(lambda s: " ".join(random.choices(s.split(), k=10))).to_list()
    )

    return ids, acronyms, list(zip(acronyms, targets))


@pytest.mark.parametrize("on_disk", [True, False])
def test_ndb_finetunable_retriever_associate(on_disk):
    db = build_db(on_disk)

    ids, acronyms, associations = get_association_samples()

    acc_before_associate = compute_accuracy(db, queries=acronyms, labels=ids)
    print("before associate p@1 =", acc_before_associate)
    assert acc_before_associate <= 0.5

    db.associate_batch(associations, retriever_strength=1)

    acc_after_associate = compute_accuracy(db, queries=acronyms, labels=ids)
    print("after associate p@1 =", acc_after_associate)
    assert acc_after_associate >= 0.9


@pytest.mark.parametrize("on_disk", [True, False])
def test_ndb_finetunable_retriever_save_load(on_disk):
    db = build_db(on_disk)

    save_path = "./finetunable_retriever.ndb"

    db.save(save_path)

    db = ndb.NeuralDB.from_checkpoint(save_path)
    shutil.rmtree(save_path)

    check_basic_query_accuracy(db)
