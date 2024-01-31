import os
import shutil
from pathlib import Path
from typing import List

import numpy as np
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
from thirdai import dataset
from thirdai import neural_db as ndb

pytestmark = [pytest.mark.unit, pytest.mark.release]


small_doc_set = [ndb.CSV(CSV_FILE), ndb.PDF(PDF_FILE, on_disk=True)]
all_local_docs = [get_doc() for get_doc in all_local_doc_getters]


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


def test_neural_db_all_methods_work_on_new_model():
    db = ndb.NeuralDB("user")
    all_methods_work(
        db,
        docs=small_doc_set,
        num_duplicate_docs=0,
        assert_acc=False,
    )


def test_neuralb_db_all_methods_work_on_new_mach_mixture():
    number_models = 2
    db = ndb.NeuralDB("user", number_models=number_models)
    all_methods_work(
        db,
        docs=small_doc_set,
        num_duplicate_docs=0,
        assert_acc=False,
    )


def test_neural_db_constrained_search_with_single_constraint():
    db = ndb.NeuralDB()
    db.insert(docs_with_meta(), train=False)
    for constraint in metadata_constraints:
        # Since we always use the same query, we know that we're getting different
        # results solely due to the imposed constraints.
        references = db.search("hello", top_k=10, constraints={"meta": constraint})
        assert len(references) > 0
        assert all([constraint == ref.metadata["meta"] for ref in references])


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
        references = db.search("hello", top_k=10, constraints=constraints)
        assert len(references) > 0
        assert all(
            [
                all([ref.metadata[key] == value for key, value in constraints.items()])
                for ref in references
            ]
        )


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


def test_neural_db_constrained_search_row_level_constraints(empty_neural_db):
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

    os.remove("ice_cream.csv")
    os.remove("pizza.csv")

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


def test_neural_db_rerank_search():
    def char4(sentence):
        return [sentence[i : i + 4] for i in range(len(sentence) - 3)]

    def custom_tokenize(sentence):
        tokens = []
        sentence = sentence.lower()
        import re

        sentence = re.sub(r"[<>=`\-,.{}:|;/@#?!&~$\[\]()\"']+\ *", " ", sentence)
        for word in sentence.split(" "):
            if len(word) > 4:
                tokens.extend(char4(word))
        return set(tokens)

    def score(query_tokens, docs_tokens):
        return len(query_tokens.intersection(docs_tokens))

    db = ndb.NeuralDB("user")
    db.insert(all_local_docs, train=False)

    query = (
        "The standard chunk of Lorem Ipsum used since the 1500s is reproduced below for"
        " those interested. Sections 1.10.32 and 1.10.33 from de Finibus Bonorum et"
        " Malorum by Cicero are also reproduced in their exact original form,"
        " accompanied by English versions from the 1914 translation by H. Rackham."
    )
    results = db.search(query, top_k=10, rerank=True)

    query_tokens = custom_tokenize(query)
    docs_tokens = [custom_tokenize(r.text) for r in results]

    for i in range(1, len(docs_tokens)):
        prev_score = score(query_tokens, docs_tokens[i - 1])
        cur_score = score(query_tokens, docs_tokens[i])
        assert prev_score >= cur_score
        assert results[i - 1].score >= results[i].score


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


def test_neural_db_reranking():
    db = ndb.NeuralDB("user")
    db.insert(all_local_docs, train=True)

    query = "Lorem Ipsum"

    # Reranking with rerank_threshold = 0 is the same as not reranking
    assert references_are_equal(
        db.search(query, top_k=100),
        db.search(query, top_k=100, rerank=True, rerank_threshold=0),
    )

    # Reranking with rerank_threshold = None or inf equals reranking everything
    assert references_are_equal(
        db.search(query, top_k=100, rerank=True, rerank_threshold=None),
        db.search(query, top_k=100, rerank=True, rerank_threshold=float("inf")),
    )

    # Results are different with and without reranking
    assert not references_are_equal(
        db.search(query, top_k=100),
        db.search(query, top_k=100, rerank=True, rerank_threshold=None),
    )

    # Assert that threshold top_k defaults to top_k
    assert references_are_equal(
        db.search(query, top_k=10, rerank=True, rerank_threshold=1.5),
        db.search(
            query, top_k=10, rerank=True, rerank_threshold=1.5, top_k_threshold=10
        ),
    )

    # Scores are in descending order with and without ranking
    base_results = db.search(query, top_k=100)
    reranked_results = db.search(query, top_k=100, rerank=True, rerank_threshold=None)
    assert descending_order([ref.score for ref in base_results])
    assert descending_order([ref.score for ref in reranked_results])
    assert reranked_results[0].score <= base_results[0].score
    assert reranked_results[-1].score >= base_results[-1].score


def test_neural_db_reranking_threshold():
    db = ndb.NeuralDB("user")
    db.insert(all_local_docs, train=True)

    query = "agreement"

    # Items with scores above the threshold are not reranked
    base_results = db.search(query, top_k=10)
    scores = np.array([ref.score for ref in base_results])
    mean_score = np.mean(scores)
    # Set threshold to 1.0 (of mean) so some of the top 10 references are
    # guaranteed to pass the threshold.
    rerank_threshold = 1.0
    threshold = rerank_threshold * mean_score
    for rerank_start, score in enumerate(scores):
        if score < threshold:
            break
    assert rerank_start > 0 and rerank_start < len(scores)
    reranked_results = db.search(
        query,
        top_k=10,
        rerank=True,
        top_k_rerank=100,
        rerank_threshold=rerank_threshold,
    )
    assert references_are_equal(
        base_results[:rerank_start], reranked_results[:rerank_start]
    )
    assert not references_are_equal(
        base_results[rerank_start:], reranked_results[rerank_start:]
    )
    assert descending_order([ref.score for ref in reranked_results])

    # Reranked order is consistent with reranker
    top_100_results = db.search(query, top_k=100)
    ranker = thirdai.dataset.KeywordOverlapRanker()
    reranked_indices, _ = ranker.rank(
        query, [ref.text for ref in top_100_results[rerank_start:]]
    )
    ranker_results = [top_100_results[rerank_start + i] for i in reranked_indices]
    assert references_are_equal(
        reranked_results[rerank_start:],
        ranker_results[: 10 - rerank_start],
        check_equal_scores=False,
    )


def test_custom_epoch(create_simple_dataset):
    db = ndb.NeuralDB(user_id="user")

    doc = ndb.CSV(
        path=create_simple_dataset,
        id_column="label",
        strong_columns=["text"],
        weak_columns=["text"],
        reference_columns=["text"],
    )

    batch_count = 0

    def count_batch(progress):
        nonlocal batch_count
        batch_count += 2  # Because progress function gets called for even batches only.

    num_epochs = 10
    db.insert(sources=[doc], epochs=num_epochs, on_progress=count_batch)

    # And number of batches in 'create_simple_dataset' is 1, so, number of epochs that the model got trained for will be number of batches.
    assert num_epochs == batch_count
