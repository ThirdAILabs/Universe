import os
import random
import shutil
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pytest
import thirdai
from ndb_utils import (
    CSV_FILE,
    PDF_FILE,
    all_local_doc_getters,
    associate_works,
    clear_sources_works,
    create_simple_dataset,
    docs_with_meta,
    empty_neural_db,
    insert_works,
    metadata_constraints,
    num_duplicate_local_doc_getters,
    num_duplicate_on_diskable_doc_getters,
    on_diskable_doc_getters,
    save_load_works,
    search_works,
    train_simple_neural_db,
    upvote_batch_works,
    upvote_works,
)
from thirdai import bolt
from thirdai import neural_db as ndb
from thirdai.neural_db.models.models import merge_results

pytestmark = [pytest.mark.unit, pytest.mark.release]


@pytest.fixture(scope="session")
def small_doc_set():
    return [ndb.CSV(CSV_FILE), ndb.PDF(PDF_FILE, on_disk=True)]


@pytest.fixture(scope="module")
def small_doc_set_dask():
    return [ndb.CSV(CSV_FILE, use_dask=True), ndb.PDF(PDF_FILE, on_disk=True)]


@pytest.fixture(scope="module")
def small_doc_set_dask_multiple_partitions():
    return [
        ndb.CSV(CSV_FILE, use_dask=True, blocksize=1e3),
        ndb.PDF(PDF_FILE, on_disk=True),
    ]


@pytest.fixture(scope="session")
def all_local_docs():
    return [get_doc() for get_doc in all_local_doc_getters]


def test_neural_db_reference_scores(train_simple_neural_db):
    db = train_simple_neural_db

    results = db.search("are apples green or red ?", top_k=10)
    for r in results:
        assert 0 <= r.score and r.score <= 1

    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def all_methods_work(
    db: ndb.NeuralDB,
    docs: List[ndb.Document],
    num_duplicate_docs: int,
    assert_acc: bool,
):
    insert_works(db, docs, num_duplicate_docs)
    search_works(db, docs, assert_acc)
    upvote_works(db)
    associate_works(db)
    save_load_works(db)
    clear_sources_works(db)


def test_neural_db_all_methods_work_finetunable_retriever(small_doc_set):
    db = ndb.NeuralDB()
    all_methods_work(
        db,
        docs=small_doc_set,
        num_duplicate_docs=0,
        assert_acc=False,
    )


@pytest.mark.parametrize("retriever", ["mach", "hybrid"])
def test_neural_db_all_methods_work_on_new_model_with_inverted(
    small_doc_set, retriever
):
    db = ndb.NeuralDB(retriever=retriever)
    all_methods_work(
        db,
        docs=small_doc_set,
        num_duplicate_docs=0,
        assert_acc=False,
    )


@pytest.mark.parametrize(
    "doc_set", ["small_doc_set_dask", "small_doc_set_dask_multiple_partitions"]
)
def test_neural_db_all_methods_work_with_dask(doc_set, request):
    db = ndb.NeuralDB()
    all_methods_work(
        db,
        docs=request.getfixturevalue(doc_set),
        num_duplicate_docs=0,
        assert_acc=False,
    )


@pytest.mark.parametrize("num_shards", [1, 2])
def test_neural_db_all_methods_work_on_new_mach_mixture(small_doc_set, num_shards):
    db = ndb.NeuralDB(
        num_shards=num_shards,
        num_models_per_shard=2,
        fhr=20_000,
        extreme_output_dim=2_000,
        retriever="hybrid",
    )
    all_methods_work(
        db,
        docs=small_doc_set,
        num_duplicate_docs=0,
        assert_acc=False,
    )


def test_neural_db_constrained_search_with_single_constraint():
    db = ndb.NeuralDB(retriever="mach")
    db.insert(docs_with_meta(), train=False)
    for constraint in metadata_constraints:
        # Since we always use the same query, we know that we're getting different
        # results solely due to the imposed constraints.
        references = db.search("hello", top_k=10, constraints={"meta": constraint})
        assert len(references) > 0
        assert all([constraint == ref.metadata["meta"] for ref in references])


@pytest.mark.parametrize("empty_neural_db", ([1, 2, "no_mach"]), indirect=True)
def test_neural_db_constrained_search_with_multiple_constraints(empty_neural_db):
    documents = [
        ndb.PDF(PDF_FILE, metadata={"language": "English", "county": "Harris"}),
        ndb.PDF(PDF_FILE, metadata={"language": "Spanish", "county": "Austin"}),
    ]
    db = empty_neural_db
    db.clear_sources()  # clear sources in case a different test added sources
    db.insert(documents, train=False)
    for constraints in [
        {"language": "English", "county": "Harris"},
        {"language": "Spanish", "county": "Austin"},
    ]:
        # Since we always use the same query, we know that we're getting different
        # results solely due to the imposed constraints.
        references = db.search(
            "Confidential Information", top_k=10, constraints=constraints
        )
        assert len(references) > 0
        assert all(
            [
                all([ref.metadata[key] == value for key, value in constraints.items()])
                for ref in references
            ]
        )


@pytest.mark.parametrize("empty_neural_db", ([1, 2, "no_mach"]), indirect=True)
def test_neural_db_constrained_search_with_list_constraints(empty_neural_db):
    documents = [
        ndb.PDF(PDF_FILE, metadata={"groups": [1], "county": "Harris"}),
        ndb.PDF(PDF_FILE, metadata={"groups": [1, 2], "county": "Austin"}),
        ndb.PDF(PDF_FILE, metadata={"groups": [3], "county": "Dallas"}),
    ]
    db = empty_neural_db
    db.clear_sources()  # clear sources in case a different test added sources
    db.insert(documents, train=False)
    for constraints in [
        {"groups": ndb.AnyOf([1])},
        {"groups": ndb.AnyOf([2])},
        {"groups": ndb.AnyOf([1, 3])},
    ]:
        references = db.search(
            "Confidential Information", top_k=10, constraints=constraints
        )
        assert len(references) > 0
        assert all(
            [
                all(
                    [
                        len(set(value.values) & set(ref.metadata[key])) > 0
                        for key, value in constraints.items()
                    ]
                )
                for ref in references
            ]
        )


@pytest.mark.parametrize("empty_neural_db", ([1, 2]), indirect=True)
def test_neural_db_constrained_search_with_multiple_constraints_multiple_models(
    empty_neural_db,
):
    documents = [
        ndb.PDF(PDF_FILE, metadata={"language": "English", "county": "Harris"}),
        ndb.PDF(PDF_FILE, metadata={"language": "Spanish", "county": "Austin"}),
    ]
    db = empty_neural_db
    db.clear_sources()  # clear sources in case a different test added sources
    db.insert(documents, train=False)
    for constraints in [
        {"language": "English", "county": "Harris"},
        {"language": "Spanish", "county": "Austin"},
    ]:
        # Since we always use the same query, we know that we're getting different
        # results solely due to the imposed constraints.
        references = db.search("hello", top_k=10, constraints=constraints)
        assert len(references) == 10
        assert all(
            [
                all([ref.metadata[key] == value for key, value in constraints.items()])
                for ref in references
            ]
        )


@pytest.mark.parametrize("empty_neural_db", ([1, 2]), indirect=True)
def test_neural_db_constrained_search_with_set_constraint(empty_neural_db):
    documents = [
        ndb.PDF(PDF_FILE, metadata={"date": "2023-10-10"}),
        ndb.PDF(PDF_FILE, metadata={"date": "2022-10-10"}),
        ndb.PDF(PDF_FILE, metadata={"date": "2021-10-10"}),
    ]
    db = empty_neural_db
    db.clear_sources()  # clear sources in case a different test added sources
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


@pytest.mark.parametrize("empty_neural_db", ([1, 2]), indirect=True)
def test_neural_db_constrained_search_with_range_constraint(empty_neural_db):
    documents = [
        ndb.PDF(PDF_FILE, metadata={"date": "2023-10-10", "score": 0.5}),
        ndb.PDF(PDF_FILE, metadata={"date": "2022-10-10", "score": 0.9}),
    ]
    db = empty_neural_db
    db.clear_sources()  # clear sources in case a different test added sources
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


@pytest.mark.parametrize("empty_neural_db", ([1, 2]), indirect=True)
def test_neural_db_constrained_search_with_comparison_constraint(empty_neural_db):
    documents = [
        ndb.PDF(PDF_FILE, metadata={"date": "2023-10-10", "score": 0.5}),
        ndb.PDF(PDF_FILE, metadata={"date": "2022-10-10", "score": 0.9}),
    ]
    db = empty_neural_db
    db.clear_sources()  # clear sources in case a different test added sources
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


@pytest.mark.parametrize("empty_neural_db", ([1, 2]), indirect=True)
def test_neural_db_constrained_search_no_matches(empty_neural_db):
    documents = [
        ndb.PDF(PDF_FILE, metadata={"date": "2023-10-10", "score": 0.5}),
    ]
    db = empty_neural_db
    db.clear_sources()  # clear sources in case a different test added sources
    db.insert(documents, train=False)

    references = db.search(
        "hello", top_k=10, constraints={"date": ndb.GreaterThan("2024-01-01")}
    )
    assert len(references) == 0


@pytest.mark.parametrize("empty_neural_db", ([1, 2]), indirect=True)
@pytest.mark.parametrize("use_dask", ([True, False]))
def test_neural_db_constrained_search_row_level_constraints(empty_neural_db, use_dask):
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
            use_dask=use_dask,
        )
    ]
    db = empty_neural_db
    db.clear_sources()  # clear sources in case a different test added sources
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


@pytest.mark.parametrize("empty_neural_db", ([1, 2, "no_mach"]), indirect=True)
def test_neural_db_delete_document(empty_neural_db):
    with open("ice_cream.csv", "w") as f:
        f.write("text,id\n")
        f.write("ice cream,0\n")

    with open("pizza.csv", "w") as f:
        f.write("text,id\n")
        f.write("pizza,0\n")

    db = empty_neural_db
    db.clear_sources()  # clear sources in case a different test added sources
    docs = [
        ndb.CSV(
            "ice_cream.csv",
            id_column="id",
            strong_columns=["text"],
            weak_columns=["text"],
            reference_columns=["text"],
            metadata={"about": "ice cream"},
            use_dask=True,
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

    db.delete([ice_cream_source_id])

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

    os.remove("ice_cream.csv")
    os.remove("pizza.csv")


def test_neural_db_delete_document_with_finetunable_retriever():
    # The other delete test is only returning 1 entity, so it will only return
    # the top result from mach, thus it doesn't test if the inverted index is
    # returning the result.
    db = ndb.NeuralDB()

    texts = [
        "apples are green",
        "bananas are yellow",
        "oranges are orange",
        "spinach is green",
        "apples are red",
    ]

    ids = db.insert(
        [ndb.InMemoryText(name=str(i), texts=[text]) for i, text in enumerate(texts)]
    )

    results = db.search(texts[-1], top_k=4)
    assert 4 in [result.id for result in results]

    db.delete([ids[-1]])

    results = db.search(texts[-1], top_k=4)
    assert 4 not in [result.id for result in results]


def references_are_equal(references_1, references_2, check_equal_scores=True):
    if len(references_1) != len(references_2):
        return False
    for ref1, ref2 in zip(references_1, references_2):
        if ref1.id != ref2.id:
            return False
        if check_equal_scores and ref1.score != ref2.score:
            return False
    return True


def descending_order(seq):
    return all(seq[i] >= seq[i + 1] for i in range(len(seq) - 1))


def test_finetunable_retriever_improves_zero_shot():
    docs = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../auto_ml/python_tests/texts.csv",
    )

    df = pd.read_csv(docs)

    queries = df["text"].map(lambda t: " ".join(random.choices(t.split(" "), k=15)))

    def compute_acc(db):
        correct = 0
        for label, q in enumerate(queries):
            results = [r.id for r in db.search(q, top_k=2)]
            if label in results:
                correct += 1

        return correct / len(queries)

    combined_db = ndb.NeuralDB(retriever="hybrid")
    combined_db.insert(
        [ndb.CSV(docs, id_column="id", weak_columns=["text"])], train=False
    )

    assert compute_acc(combined_db) > 0.9

    mach_only_db = ndb.NeuralDB(retriever="mach")
    mach_only_db.insert(
        [ndb.CSV(docs, id_column="id", weak_columns=["text"])], train=False
    )

    assert compute_acc(mach_only_db) < 0.1


def test_neural_db_retriever_specification():
    db = ndb.NeuralDB(retriever="hybrid")

    texts = [
        "apples are green",
        "bananas are yellow",
        "oranges are orange",
        "spinach is green",
        "apples are red",
        "grapes are purple",
        "lemons are yellow",
        "limes are green",
        "carrots are orange",
        "celery is green",
    ]

    db.insert(
        [ndb.InMemoryText(name=str(i), texts=[text]) for i, text in enumerate(texts)],
        train=False,
    )

    combined = set(ref.retriever for ref in db.search("carrots bananas", top_k=10))
    assert "mach" in combined

    mach_results = db.search("carrots bananas", top_k=10, retriever="mach")
    assert len(mach_results) > 0
    for res in mach_results:
        assert res.retriever == "mach"

    index_results = db.search(
        "carrots bananas", top_k=10, retriever="finetunable_retriever"
    )
    assert len(index_results) > 0
    for res in index_results:
        assert res.retriever == "finetunable_retriever"


def test_result_merging():
    results_a = [
        (1, 5.0, "a"),
        (2, 4.0, "a"),
        (3, 3.0, "a"),
        (4, 2.0, "a"),
        (6, 1.0, "a"),
    ]
    results_b = [
        (2, 5.0, "b"),
        (7, 4.0, "b"),
        (3, 3.0, "b"),
        (5, 2.0, "b"),
        (4, 1.0, "b"),
    ]

    expected_output = [1, 2, 7, 3, 4, 5, 6]

    assert [x[0] for x in merge_results(results_a, results_b, k=10)] == expected_output


def test_insert_callback(small_doc_set):
    db = ndb.NeuralDB(user_id="test_coldstart_callback", retriever="mach")

    class EpochCheck(bolt.train.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.epochs_completed = 0

        def on_epoch_end(self):
            self.epochs_completed += 1

    epoch_count_callback = EpochCheck()

    epochs = 6
    db.insert(small_doc_set, epochs=epochs, callbacks=[epoch_count_callback])

    assert epoch_count_callback.epochs_completed == epochs


def test_neural_db_prioritizes_finetunable_retriever_results():
    db = ndb.NeuralDB(retriever="hybrid")

    texts = [
        "apples are green",
        "bananas are yellow",
        "oranges are orange",
        "spinach is green",
        "apples are red",
    ]

    db.insert(
        [ndb.InMemoryText(name=str(i), texts=[text]) for i, text in enumerate(texts)],
        train=False,
    )

    assert db.search("carrots bananas", top_k=5)[0].retriever == "finetunable_retriever"
    assert db.search("carrots bananas", top_k=5, mach_first=True)[0].retriever == "mach"
