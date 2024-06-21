import numpy as np
import pytest
import thirdai
from ndb_utils import create_simple_dataset
from thirdai import neural_db as ndb
from thirdai.neural_db.models.mach_defaults import autotune_from_scratch_min_max_epochs

from .test_neural_db import descending_order, references_are_equal

pytestmark = [pytest.mark.unit]


def test_custom_epoch(create_simple_dataset):
    db = ndb.NeuralDB(user_id="user", retriever="hybrid")

    doc = ndb.CSV(
        path=create_simple_dataset,
        id_column="label",
        strong_columns=["text"],
        weak_columns=["text"],
        reference_columns=["text"],
    )

    num_epochs = 10
    db.insert(sources=[doc], epochs=num_epochs)

    assert num_epochs == db._savable_state.model.get_model()._get_model().epochs()


def test_neuraldb_stopping_condition(create_simple_dataset):
    db = ndb.NeuralDB(user_id="user", retriever="hybrid")

    doc = ndb.CSV(
        path=create_simple_dataset,
        id_column="label",
        strong_columns=["text"],
        weak_columns=["text"],
        reference_columns=["text"],
    )

    db.insert(sources=[doc])

    min_epochs, _ = autotune_from_scratch_min_max_epochs(size=1)

    # Our training stops when epochs >= min_epochs and accuracy >= 0.95
    # Since there is only 1 sample in the CSV the db should stop at min_epochs
    model_epoch_count = db._savable_state.model.get_model()._get_model().epochs()
    assert min_epochs == model_epoch_count - 1


def test_neural_db_reranking(all_local_docs):
    db = ndb.NeuralDB("user", retriever="mach")
    db.insert(all_local_docs, train=True, epochs=1)

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


def test_neural_db_reranking_threshold(all_local_docs):
    db = ndb.NeuralDB("user", retriever="mach")
    db.insert(all_local_docs, train=True, epochs=1)

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
        reranker="lexical",
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


def test_neural_db_rerank_search(all_local_docs):
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
    results = db.search(query, top_k=10, rerank=True, reranker="lexical")

    query_tokens = custom_tokenize(query)
    docs_tokens = [custom_tokenize(r.text) for r in results]

    for i in range(1, len(docs_tokens)):
        prev_score = score(query_tokens, docs_tokens[i - 1])
        cur_score = score(query_tokens, docs_tokens[i])
        assert prev_score >= cur_score
        assert results[i - 1].score >= results[i].score
