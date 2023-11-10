import os
import shutil
from pathlib import Path
from typing import List

import pytest
from ndb_utils import (
    PDF_FILE,
    all_local_doc_getters,
    create_simple_dataset,
    docs_with_meta,
    metadata_constraints,
    train_simple_neural_db,
)
from thirdai import bolt
from thirdai import neural_db as ndb

pytestmark = [pytest.mark.unit, pytest.mark.release]


def test_neural_db_reference_scores(train_simple_neural_db):
    db = train_simple_neural_db

    results = db.search("are apples green or red ?", top_k=10)
    for r in results:
        assert 0 <= r.score and r.score <= 1

    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def db_from_bazaar():
    bazaar = ndb.Bazaar(cache_dir=".")
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
    assert len(db.sources()) == 9

    initial_scores = [r.score for r in db.search(ARBITRARY_QUERY, top_k=5)]

    db.insert(docs, train=True)
    assert len(db.sources()) == 9

    assert [r.score for r in db.search(ARBITRARY_QUERY, top_k=5)] != initial_scores


def search_works(db: ndb.NeuralDB, docs: List[ndb.Document], assert_acc: bool):
    top_k = 5
    correct_result = 0
    correct_source = 0
    for doc in docs:
        if isinstance(doc, ndb.SharePoint):
            continue
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
    assert db.sources().keys() == new_db.sources().keys()
    assert [doc.name for doc in db.sources().values()] == [
        doc.name for doc in new_db.sources().values()
    ]

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


def test_neural_db_all_methods_work_on_new_model():
    db = ndb.NeuralDB("user")
    all_docs = [get_doc() for get_doc in all_local_doc_getters]
    all_methods_work(db, all_docs, assert_acc=False)


def test_neural_db_all_methods_work_on_loaded_bazaar_model():
    db = db_from_bazaar()
    all_docs = [get_doc() for get_doc in all_local_doc_getters]
    all_methods_work(db, all_docs, assert_acc=True)


def train_model_for_supervised_training_test(model_id_delimiter):
    db = ndb.NeuralDB("user", id_delimiter=model_id_delimiter)

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
def test_neural_db_supervised_training_multilabel_csv(model_id_delimiter):
    db, source_ids = train_model_for_supervised_training_test(model_id_delimiter)

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

    db.supervised_train([sup_doc], learning_rate=0.1, epochs=20)

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

    db.supervised_train([sup_doc], learning_rate=0.1, epochs=20)

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


def test_neural_db_constrained_search_with_single_constraint():
    db = ndb.NeuralDB()
    db.insert(docs_with_meta(), train=False)
    for constraint in metadata_constraints:
        # Since we always use the same query, we know that we're getting different
        # results solely due to the imposed constraints.
        references = db.search("hello", top_k=10, constraints={"meta": constraint})
        assert len(references) > 0
        assert all([constraint == ref.metadata["meta"] for ref in references])


def test_neural_db_constrained_search_with_multiple_constraints():
    documents = [
        ndb.PDF(PDF_FILE, metadata={"language": "English", "county": "Harris"}),
        ndb.PDF(PDF_FILE, metadata={"language": "Spanish", "county": "Austin"}),
    ]
    db = ndb.NeuralDB()
    db.insert(documents, train=False)
    for constraints in [
        {"language": "English", "county": "Harris"},
        {"language": "Spanish", "county": "Austin"},
    ]:
        # Since we always use the same query, we know that we're getting different
        # results solely due to the imposed constraints.
        references = db.search("hello", top_k=10, constraints=constraints)
        assert len(references) > 0
        assert all(
            [
                all([ref.metadata[key] == value for key, value in constraints.items()])
                for ref in references
            ]
        )


def test_neural_db_constrained_search_with_set_constraint():
    documents = [
        ndb.PDF(PDF_FILE, metadata={"date": "2023-10-10"}),
        ndb.PDF(PDF_FILE, metadata={"date": "2022-10-10"}),
        ndb.PDF(PDF_FILE, metadata={"date": "2021-10-10"}),
    ]
    db = ndb.NeuralDB()
    db.insert(documents, train=False)

    references = db.search(
        "hello",
        top_k=20,
        # Include 1923-10-10 to make sure it doesnt break if none of the documents
        # match the constraints.
        constraints={"date": ndb.AnyOf(["2023-10-10", "2022-10-10", "1923-10-10"])},
    )
    assert len(references) > 0
    assert all(
        [
            ref.metadata["date"] == "2023-10-10" or ref.metadata["date"] == "2022-10-10"
            for ref in references
        ]
    )

    # Make sure that the other document shows up if we don't constrain the search.
    references = db.search("hello", top_k=20)
    assert any([ref.metadata["date"] == "2021-10-10" for ref in references])


def test_neural_db_constrained_search_with_range_constraint():
    documents = [
        ndb.PDF(PDF_FILE, metadata={"date": "2023-10-10", "score": 0.5}),
        ndb.PDF(PDF_FILE, metadata={"date": "2022-10-10", "score": 0.9}),
    ]
    db = ndb.NeuralDB()
    db.insert(documents, train=False)

    # Make sure that without constraints, we get results from both documents.
    references = db.search("hello", top_k=10)
    assert len(references) > 0
    assert not all([ref.metadata["date"] == "2023-10-10" for ref in references])

    references = db.search(
        "hello", top_k=10, constraints={"date": ndb.InRange("2023-01-01", "2023-12-31")}
    )
    assert len(references) > 0
    assert all([ref.metadata["date"] == "2023-10-10" for ref in references])

    references = db.search(
        "hello", top_k=10, constraints={"score": ndb.InRange(0.6, 1.0)}
    )
    assert len(references) > 0
    assert all([ref.metadata["score"] == 0.9 for ref in references])


def test_neural_db_constrained_search_with_comparison_constraint():
    documents = [
        ndb.PDF(PDF_FILE, metadata={"date": "2023-10-10", "score": 0.5}),
        ndb.PDF(PDF_FILE, metadata={"date": "2022-10-10", "score": 0.9}),
    ]
    db = ndb.NeuralDB()
    db.insert(documents, train=False)

    # Make sure that without constraints, we get results from both documents.
    references = db.search("hello", top_k=10)
    assert len(references) > 0
    assert not all([ref.metadata["date"] == "2023-10-10" for ref in references])

    references = db.search(
        "hello", top_k=10, constraints={"date": ndb.GreaterThan("2023-01-01")}
    )
    assert len(references) > 0
    assert all([ref.metadata["date"] == "2023-10-10" for ref in references])

    references = db.search("hello", top_k=10, constraints={"score": ndb.LessThan(0.6)})
    assert len(references) > 0
    assert all([ref.metadata["score"] == 0.5 for ref in references])


def test_neural_db_constrained_search_no_matches():
    documents = [
        ndb.PDF(PDF_FILE, metadata={"date": "2023-10-10", "score": 0.5}),
    ]
    db = ndb.NeuralDB()
    db.insert(documents, train=False)

    references = db.search(
        "hello", top_k=10, constraints={"date": ndb.GreaterThan("2024-01-01")}
    )
    assert len(references) == 0


def test_neural_db_constrained_search_row_level_constraints():
    csv_contents = [
        "id,text,date",
    ] + [f"{i},a reusable chunk of text,{1950 + i}-10-10" for i in range(100)]

    with open("chunks.csv", "w") as o:
        for line in csv_contents:
            o.write(line + "\n")

    documents = [
        ndb.CSV(
            "chunks.csv",
            id_column="id",
            strong_columns=["text"],
            weak_columns=["text"],
            reference_columns=["text"],
            index_columns=["date"],
        )
    ]
    db = ndb.NeuralDB()
    db.insert(documents, train=True)

    references = db.search(
        "a reusable chunk of text",
        top_k=52,
        constraints={"date": ndb.GreaterThan("2000-01-01")},
    )
    assert len(references) > 0
    assert all([r.metadata["date"] > "2000-01-01" for r in references])

    references = db.search("a reusable chunk of text", top_k=52)
    assert any([r.metadata["date"] < "2000-01-01" for r in references])
    assert any([r.metadata["date"] > "2000-01-01" for r in references])


def test_neural_db_delete_document():
    with open("ice_cream.csv", "w") as f:
        f.write("text,id\n")
        f.write("ice cream,0\n")

    with open("pizza.csv", "w") as f:
        f.write("text,id\n")
        f.write("pizza,0\n")

    db = ndb.NeuralDB()
    docs = [
        ndb.CSV(
            "ice_cream.csv",
            id_column="id",
            strong_columns=["text"],
            weak_columns=["text"],
            reference_columns=["text"],
            metadata={"about": "ice cream"},
        ),
        ndb.CSV(
            "pizza.csv",
            id_column="id",
            strong_columns=["text"],
            weak_columns=["text"],
            reference_columns=["text"],
            metadata={"about": "pizza"},
        ),
    ]

    for _ in range(5):
        [ice_cream_source_id, _] = db.insert(docs, train=True)

    # We will delete the ice cream file. To know that we successfully deleted
    # it, make sure it comes up as a search result before deleting, and does not
    # come up after deleting.
    result = db.search("ice cream", top_k=1)[0]
    assert result.text == "text: ice cream"
    ice_cream_id = result.id

    result = db.search("ice cream", top_k=1, constraints={"about": "ice cream"})[0]
    assert result.text == "text: ice cream"

    db.delete(ice_cream_source_id)

    results = db.search("ice cream", top_k=1)
    # pizza may not come up, so check if we got any result at all.
    if len(results) > 0:
        assert results[0].text != "text: ice cream"

    results = db.search("ice cream", top_k=1, constraints={"about": "ice cream"})
    assert len(results) == 0

    # Make sure the other document is unaffected
    result = db.search("pizza", top_k=1)[0]
    assert result.text == "text: pizza"
    pizza_id = result.id

    result = db.search("pizza", top_k=1, constraints={"about": "pizza"})[0]
    assert result.text == "text: pizza"

    # Make sure there are no problems with reinserting deleted document.
    for _ in range(5):
        db.insert(docs, train=True)
    new_ice_cream_result = db.search("ice cream", top_k=1)[0]
    assert new_ice_cream_result.text == "text: ice cream"
    assert new_ice_cream_result.id != ice_cream_id
    new_pizza_result = db.search("pizza", top_k=1)[0]
    assert new_pizza_result.text == "text: pizza"
    assert new_pizza_result.id == pizza_id

    # Make sure constrained search index is also updated
    result = db.search("ice cream", top_k=1, constraints={"about": "ice cream"})[0]
    assert result.text == "text: ice cream"


def char4(sentence):
    return [sentence[i : i + 4] for i in range(0, len(sentence) - 3, 2)]


def custom_tokenize(sentence):
    tokens = []
    sentence = sentence.lower()
    for word in sentence.split(" "):
        if len(word) > 4:
            tokens.extend(char4(word))
    return set(tokens)


def test_neural_db_rerank_search():
    db = ndb.NeuralDB("user")
    all_docs = [get_doc() for get_doc in all_local_doc_getters]
    db.insert(all_docs, train=False)

    query = "The standard chunk of Lorem Ipsum used since the 1500s is reproduced below for those interested. Sections 1.10.32 and 1.10.33 from de Finibus Bonorum et Malorum by Cicero are also reproduced in their exact original form, accompanied by English versions from the 1914 translation by H. Rackham."
    results = db.search(query, top_k=10, rerank=True)

    query_tokens = custom_tokenize(query)
    docs_tokens = [custom_tokenize(r.text) for r in results]

    first_ranked_score = len(query_tokens.intersection(docs_tokens[0]))
    last_ranked_score = len(query_tokens.intersection(docs_tokens[-1]))

    assert first_ranked_score > last_ranked_score
