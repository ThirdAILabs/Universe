import pytest
import thirdai
from doc_retrieval_helpers import get_build_and_run_functions_random
from doc_retrieval_helpers import get_build_and_run_functions_restful


@pytest.mark.unit
def test_random_docs():
    index_func, query_func = get_build_and_run_functions_random()
    index = index_func()
    query_func(index)


@pytest.mark.unit
def test_random_docs_serialization():
    index_func, query_func = get_build_and_run_functions_random()
    index1 = index_func()
    index1.serialize_to_file("test.serialized")
    index2 = thirdai.search.DocRetrieval.deserialize_from_file("test.serialized")
    query_results1 = query_func(index1)
    query_results2 = query_func(index2)
    for a, b in zip(query_results1, query_results2):
        assert a == b


@pytest.mark.unit
def test_restful():
    index_func, query_func = get_build_and_run_functions_restful()
    index = index_func()
    query_func(index)


@pytest.mark.unit
def test_restful_serialization():
    index_func, query_func = get_build_and_run_functions_restful()
    index1 = index_func()
    index1.serialize_to_file("test.serialized")
    index2 = thirdai.search.DocRetrieval.deserialize_from_file("test.serialized")
    query_results1 = query_func(index1)
    query_results2 = query_func(index2)
    for a, b in zip(query_results1, query_results2):
        assert a == b


def expect_error_on_construction(
    num_tables=1, dense_input_dimension=1, hashes_per_table=1, centroids=[[0]]
):
    with pytest.raises(Exception):
        thirdai.search.DocRetrieval(
            centroids=centroids,
            hashes_per_table=hashes_per_table,
            num_tables=num_tables,
            dense_input_dimension=dense_input_dimension,
        )


@pytest.mark.unit
def test_error_inputs():
    import time

    start = time.time()
    expect_error_on_construction(num_tables=0)
    expect_error_on_construction(num_tables=-7)
    expect_error_on_construction(dense_input_dimension=0)
    expect_error_on_construction(dense_input_dimension=-7)
    expect_error_on_construction(
        dense_input_dimension=100
    )  # Since the default centroids will still be dim=1
    expect_error_on_construction(hashes_per_table=0)
    expect_error_on_construction(hashes_per_table=-7)
    expect_error_on_construction(centroids=[])
    end = time.time()

    # We have a time assertion because should catch input errors quickly and
    # not e.g. build any big objects with invalid input
    assert end - start < 0.01


# TOOD(josh): Add the following tests:
# 1. Test with lots of elements to ensure centroid reranking is reasonably fast
# 2. Test to ensure that finding centroids in C++ is as fast as in numpy
# (will need Eigen or similar
# blas library in C++ for this)
# 3. Test with overall benchmarking test, probably on blade node,
#    perhaps download msmarco index
