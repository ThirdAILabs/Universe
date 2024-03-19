import os
import random
from collections import defaultdict

import pytest
from thirdai import neural_db as ndb

pytestmark = [pytest.mark.unit, pytest.mark.release]


def get_label_from_same_shard(db: ndb.NeuralDB, original_label: int, number_labels):
    """
    Supervised Training Tests for a mixture of mach models requires that the new label associated with the query is in the same shard as the original label.

    For example,
    number_models = 2
    labels in model 1 : label_1 = {0,1,2}
    label_2 = {3,4,5}

    If original data had (query, label) (q,0), to test supervised training, we should have new label l from the set {1,2}.
    This is because if l is in {3,4,5} then model 1 will always predict 0 with high activation and hence, irrespective of how long we train, there's no guarantee that activations of any of 3,4,5 can be as high as 0 while predicting for query q.

    Hence, we return labels from the same shard for supervised training.
    """
    label_to_segment_map = db._savable_state.model.label_to_segment_map
    segment_to_label_map = defaultdict(list)
    for label in label_to_segment_map:
        segment_to_label_map[label_to_segment_map[label][0]].append(label)

    filtered_labels = [
        l
        for l in segment_to_label_map[label_to_segment_map[original_label][0]]
        if l != original_label
    ]

    assert number_labels <= len(filtered_labels)
    return random.sample(filtered_labels, number_labels)


def train_model_for_supervised_training_test(model_id_delimiter, num_shards=1, num_models_per_shard=1):
    db = ndb.NeuralDB(
        "user", id_delimiter=model_id_delimiter, num_shards=num_shards, num_models_per_shard=num_models_per_shard,
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


@pytest.mark.parametrize("model_id_delimiter", [" ", None])
@pytest.mark.parametrize("num_shards", [1,2])
def test_neural_db_supervised_training_mixture(model_id_delimiter, num_shards):
    db, _ = train_model_for_supervised_training_test(
        model_id_delimiter=model_id_delimiter, num_shards=num_shards, num_models_per_shard=2,
    )
    queries = ["first", "sixth"]
    new_labels = [
        get_label_from_same_shard(db, original_label=0, number_labels=1),
        get_label_from_same_shard(db, original_label=5, number_labels=2),
    ]

    db.supervised_train(
        [ndb.Sup(queries=queries, labels=new_labels, uses_db_id=True)],
        learning_rate=0.001,
        epochs=50,
    )

    assert db.search(queries[0], top_k=1)[0].id == new_labels[0][0]
    expect_top_2_results(db, queries[1], new_labels[1])


@pytest.mark.parametrize("model_id_delimiter", [" ", None])
def test_neural_db_supervised_training_multilabel_csv(model_id_delimiter):
    db, source_ids = train_model_for_supervised_training_test(model_id_delimiter)

    with open("mock_sup_1.csv", "w") as out:
        out.write("id,query\n")
        # make sure that single label rows are also handled correctly in a
        # multilabel dataset.
        out.write("4,first\n")
        out.write("0:1,fifth\n")
        out.write("2:3:,ninth\n")

    sup_doc = ndb.Sup(
        "mock_sup_1.csv",
        query_column="query",
        id_column="id",
        id_delimiter=":",
        source_id=source_ids[0],
    )

    db.supervised_train([sup_doc], learning_rate=0.01, epochs=20)

    assert db.search("first", top_k=1)[0].id == 4
    expect_top_2_results(db, "fifth", [0, 1])
    expect_top_2_results(db, "ninth", [2, 3])

    with open("mock_sup_2.csv", "w") as out:
        out.write("id,query\n")
        # make sure that single label rows are also handled correctly in a
        # multilabel dataset.
        out.write("4,second\n")
        out.write("0:1,tenth\n")
        out.write("2:3:,fifth\n")

    sup_doc = ndb.Sup(
        "mock_sup_2.csv",
        query_column="query",
        id_column="id",
        id_delimiter=":",
        source_id=source_ids[1],
    )

    db.supervised_train([sup_doc], learning_rate=0.01, epochs=20)

    assert db.search("second", top_k=1)[0].id == 9
    expect_top_2_results(db, "tenth", [5, 6])
    expect_top_2_results(db, "fifth", [7, 8])

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


def test_neural_db_supervised_train_with_comma():
    db = ndb.NeuralDB()

    with open("mock_unsup.csv", "w") as f:
        f.write("id,strong\n")
        f.write('0,"first, second"\n')

    source_ids = db.insert(
        [ndb.CSV("mock_unsup.csv", id_column="id", strong_columns=["strong"])]
    )

    with open("mock_sup.csv", "w") as f:
        f.write("id,query\n")
        f.write('0,"sixth, seventh"\n')

    sup_doc = ndb.Sup(
        "mock_sup.csv",
        query_column="query",
        id_column="id",
        source_id=source_ids[0],
    )

    db.supervised_train([sup_doc], learning_rate=0.1, epochs=1)
