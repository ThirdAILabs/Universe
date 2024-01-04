import os
from itertools import product

import pytest
from thirdai import neural_db as ndb

pytestmark = [pytest.mark.unit, pytest.mark.release]


def train_model_for_supervised_training_test(model_id_delimiter, number_models=1):
    db = ndb.NeuralDB(
        "user", id_delimiter=model_id_delimiter, number_models=number_models
    )

    with open("mock_unsup_1.csv", "w") as out:
        out.write("id,strong\n")
        out.write("0,first\n")
        out.write("1,second\n")
        out.write("2,third\n")
        out.write("3,fourth\n")
        out.write("4,fifth\n")

    with open("mock_unsup_2.csv", "w") as out:
        out.write("id,strong\n")
        out.write("0,sixth\n")
        out.write("1,seventh\n")
        out.write("2,eighth\n")
        out.write("3,ninth\n")
        out.write("4,tenth\n")

    def overfit():
        if not db.ready_to_search():
            return False
        queries = [
            "first",
            "second",
            "third",
            "fourth",
            "fifth",
            "sixth",
            "seventh",
            "eighth",
            "ninth",
            "tenth",
        ]
        for query, label in zip(queries, range(10)):
            if db.search(query, top_k=1)[0].id != label:
                return False
        return True

    while not overfit():
        source_ids = db.insert(
            [
                ndb.CSV("mock_unsup_1.csv", id_column="id", strong_columns=["strong"]),
                ndb.CSV("mock_unsup_2.csv", id_column="id", strong_columns=["strong"]),
            ]
        )

    # It is fine to remove these files since we've loaded it in memory.
    os.remove("mock_unsup_1.csv")
    os.remove("mock_unsup_2.csv")

    return db, source_ids


def expect_top_2_results(db, query, expected_results):
    result_ids = set([ref.id for ref in db.search(query, top_k=2)])
    assert len(result_ids.intersection(set(expected_results))) >= 1


@pytest.mark.parametrize("model_id_delimiter, number_models", product([" "], [2]))
def test_neural_db_supervised_training_multilabel_csv(
    model_id_delimiter, number_models
):
    print(model_id_delimiter)
    print(number_models)
    db, source_ids = train_model_for_supervised_training_test(
        model_id_delimiter, number_models=number_models
    )

    with open("mock_sup_1.csv", "w") as out:
        out.write("id,query\n")
        # make sure that single label rows are also handled correctly in a
        # multilabel dataset.
        out.write("4,first\n")
        out.write("0:1,fourth\n")
        out.write("2:3:,second\n")

    sup_doc = ndb.Sup(
        "mock_sup_1.csv",
        query_column="query",
        id_column="id",
        id_delimiter=":",
        source_id=source_ids[0],
    )

    db.supervised_train([sup_doc], learning_rate=0.002, epochs=1000)

    assert db.search("first", top_k=1)[0].id == 4
    expect_top_2_results(db, "fourth", [0, 1])
    expect_top_2_results(db, "second", [2, 3])

    with open("mock_sup_2.csv", "w") as out:
        out.write("id,query\n")
        # make sure that single label rows are also handled correctly in a
        # multilabel dataset.
        out.write("4,sixth\n")
        out.write("0:1,ninth\n")
        out.write("2:3:,seventh\n")

    sup_doc = ndb.Sup(
        "mock_sup_2.csv",
        query_column="query",
        id_column="id",
        id_delimiter=":",
        source_id=source_ids[1],
    )

    db.supervised_train([sup_doc], learning_rate=0.002, epochs=1000)

    search_res = db.search("sixth", top_k=5)
    for res in search_res:
        print(res.id, res.score)

    assert db.search("sixth", top_k=1)[0].id == 9
    expect_top_2_results(db, "ninth", [5, 6])
    expect_top_2_results(db, "seventh", [7, 8])

    os.remove("mock_sup_1.csv")
    os.remove("mock_sup_2.csv")


@pytest.mark.parametrize("model_id_delimiter", [" ", None])
def test_neural_db_supervised_training_singlelabel_csv(model_id_delimiter):
    db, source_ids = train_model_for_supervised_training_test(model_id_delimiter)

    with open("mock_sup_1.csv", "w") as out:
        out.write("id,query\n")
        out.write("4,first\n")
        out.write("0,fourth\n")
        out.write("2,second\n")

    sup_doc = ndb.Sup(
        "mock_sup_1.csv",
        query_column="query",
        id_column="id",
        source_id=source_ids[0],
    )

    db.supervised_train([sup_doc], learning_rate=0.1, epochs=20)

    assert db.search("first", top_k=1)[0].id == 4
    assert db.search("fourth", top_k=1)[0].id == 0
    assert db.search("second", top_k=1)[0].id == 2

    with open("mock_sup_2.csv", "w") as out:
        out.write("id,query\n")
        # make sure that single label rows are also handled correctly in a
        # multilabel dataset.
        out.write("4,sixth\n")
        out.write("0,ninth\n")
        out.write("2,seventh\n")

    sup_doc = ndb.Sup(
        "mock_sup_2.csv",
        query_column="query",
        id_column="id",
        source_id=source_ids[1],
    )

    db.supervised_train([sup_doc], learning_rate=0.1, epochs=20)

    assert db.search("sixth", top_k=1)[0].id == 9
    assert db.search("ninth", top_k=1)[0].id == 5
    assert db.search("seventh", top_k=1)[0].id == 7

    os.remove("mock_sup_1.csv")
    os.remove("mock_sup_2.csv")


@pytest.mark.parametrize("model_id_delimiter", [" ", None])
def test_neural_db_supervised_training_sequence_input(model_id_delimiter):
    db, source_ids = train_model_for_supervised_training_test(model_id_delimiter)

    db.supervised_train(
        [
            ndb.Sup(
                queries=["first", "fourth", "second"],
                labels=[[4], [0, 1], [2, 3]],
                source_id=source_ids[0],
            )
        ],
        learning_rate=0.1,
        epochs=20,
    )

    assert db.search("first", top_k=1)[0].id == 4
    expect_top_2_results(db, "fourth", [0, 1])
    expect_top_2_results(db, "second", [2, 3])

    db.supervised_train(
        [
            ndb.Sup(
                queries=["sixth", "ninth", "seventh"],
                labels=[[4], [0, 1], [2, 3]],
                source_id=source_ids[1],
            )
        ],
        learning_rate=0.1,
        epochs=20,
    )

    assert db.search("sixth", top_k=1)[0].id == 9
    expect_top_2_results(db, "ninth", [5, 6])
    expect_top_2_results(db, "seventh", [7, 8])


@pytest.mark.parametrize("model_id_delimiter", [" ", None])
def test_neural_db_ref_id_supervised_training_multilabel_csv(model_id_delimiter):
    db, _ = train_model_for_supervised_training_test(model_id_delimiter)

    with open("mock_sup.csv", "w") as out:
        out.write("id,query\n")
        # make sure that single label rows are also handled correctly in a
        # multilabel dataset.
        out.write("4,first\n")
        out.write("0:1,fourth\n")
        out.write("8:9:,second\n")

    db.supervised_train_with_ref_ids(
        "mock_sup.csv",
        query_column="query",
        id_column="id",
        id_delimiter=":",
        learning_rate=0.1,
        epochs=20,
    )

    assert db.search("first", top_k=1)[0].id == 4
    expect_top_2_results(db, "fourth", [0, 1])
    expect_top_2_results(db, "second", [8, 9])

    os.remove("mock_sup.csv")


@pytest.mark.parametrize("model_id_delimiter", [" ", None])
def test_neural_db_ref_id_supervised_training_singlelabel_csv(model_id_delimiter):
    db, _ = train_model_for_supervised_training_test(model_id_delimiter)

    with open("mock_sup.csv", "w") as out:
        out.write("id,query\n")
        out.write("4,first\n")
        out.write("0,fourth\n")
        out.write("8,second\n")

    db.supervised_train_with_ref_ids(
        "mock_sup.csv",
        query_column="query",
        id_column="id",
        learning_rate=0.1,
        epochs=20,
    )

    assert db.search("first", top_k=1)[0].id == 4
    assert db.search("fourth", top_k=1)[0].id == 0
    assert db.search("second", top_k=1)[0].id == 8

    os.remove("mock_sup.csv")


@pytest.mark.parametrize("model_id_delimiter", [" ", None])
def test_neural_db_ref_id_supervised_training_sequence_input(model_id_delimiter):
    db, source_ids = train_model_for_supervised_training_test(model_id_delimiter)

    db.supervised_train_with_ref_ids(
        queries=["first", "fourth", "second"],
        labels=[[4], [0, 1], [8, 9]],
        learning_rate=0.1,
        epochs=20,
    )

    assert db.search("first", top_k=1)[0].id == 4
    expect_top_2_results(db, "fourth", [0, 1])
    expect_top_2_results(db, "second", [8, 9])
    assert set([ref.id for ref in db.search("fourth", top_k=2)]) == set([0, 1])
    assert set([ref.id for ref in db.search("second", top_k=2)]) == set([8, 9])
