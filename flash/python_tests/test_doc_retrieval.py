import pytest
import thirdai
from doc_retrieval_helpers import get_build_and_run_functions_random
from doc_retrieval_helpers import get_build_and_run_functions_restful


@pytest.mark.unit
def test_doc_retrieval_random():
    index_func, query_func = get_build_and_run_functions_random()
    index = index_func()
    query_func(index)


@pytest.mark.unit
def test_doc_retrieval_random_serialization():
    index_func, query_func = get_build_and_run_functions_random()
    index1 = index_func()
    index1.serialize_to_file("test.serialized")
    index2 = thirdai.search.doc_retrieval_index.deserialize_from_file("test.serialized")
    query_results1 = query_func(index1)
    query_results2 = query_func(index2)
    for a, b in zip(query_results1, query_results2):
        assert a == b


@pytest.mark.unit
def test_doc_retrieval_restful():
    index_func, query_func = get_build_and_run_functions_restful()
    index = index_func()
    query_func(index)


@pytest.mark.unit
def test_doc_retrieval_restful_serialization():
    index_func, query_func = get_build_and_run_functions_restful()
    index1 = index_func()
    index1.serialize_to_file("test.serialized")
    index2 = thirdai.search.doc_retrieval_index.deserialize_from_file("test.serialized")
    query_results1 = query_func(index1)
    query_results2 = query_func(index2)
    for a, b in zip(query_results1, query_results2):
        assert a == b
