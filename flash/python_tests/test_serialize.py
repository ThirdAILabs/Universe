import thirdai
import pytest
import numpy as np
import random


@pytest.mark.unit
def test_doc_retrieval():

    data_dim = 100
    total_num_words = 1000
    num_docs = 1000
    words_per_doc = 200
    num_queries = 10
    words_per_query_random = 5
    words_per_query_from_doc = 10
    words_per_query = words_per_query_random + words_per_query_from_doc
    between_word_std = 1
    within_word_std = 0.1

    np.random.seed(42)
    random.seed(42)

    # General idea is each word is a normal distribution somehwhere in the space.
    # A doc is made up of a vector from each of those normal distributions. A
    # ground truth query is made up of some words from a single doc's word
    # distributions and some random words.

    # Generate word centers
    word_centers = np.random.normal(
        size=(total_num_words, data_dim), scale=between_word_std
    )

    # Generates docs
    doc_word_ids = [
        random.sample(range(total_num_words), words_per_doc) for _ in range(num_docs)
    ]
    doc_offsets = np.random.normal(
        size=(num_docs, words_per_doc, data_dim), scale=within_word_std
    )
    docs = []
    for i in range(num_docs):
        doc = []
        for j in range(words_per_doc):
            doc.append(doc_offsets[i][j] + word_centers[doc_word_ids[i][j]])
        docs.append(doc)

    # Generate queries. GT for query i is doc i
    query_random_word_ids = [
        random.sample(range(total_num_words), words_per_query_random)
        for _ in range(num_queries)
    ]
    query_same_word_ids = [
        ids[:words_per_query_from_doc] for ids in doc_word_ids[:num_queries]
    ]
    query_word_ids = [a + b for a, b in zip(query_same_word_ids, query_random_word_ids)]
    query_offsets = np.random.normal(
        size=(num_queries, words_per_query, data_dim), scale=within_word_std
    )
    queries = []
    for i in range(num_queries):
        query = []
        for j in range(words_per_query):
            query.append(query_offsets[i][j] + word_centers[query_word_ids[i][j]])
        queries.append(query)

    # Build index
    index = thirdai.search.doc_retrieval_index(
        hashes_per_table=7,
        num_tables=32,
        dense_input_dimension=data_dim,
    )
    for doc in docs:
        index.add_document(document_embeddings=np.array(doc))

    # Query index
    for gt, query in enumerate(queries):
        ranking = index.rank_documents(
            query_embeddings=np.array(query),
            document_ids_to_rank=np.array(range(num_docs)),
        )
        assert gt == ranking[0]

    index.serialize_to_file("maxflash.serialized")
    index2 = thirdai.search.doc_retrieval_index.deserialize_from_file(
        "maxflash.serialized"
    )

    print(index2)
