import pytest
import thirdai
from doc_retrieval_helpers import get_build_and_run_functions


@pytest.mark.unit
def test_doc_retrieval():
    index_func, query_func = get_build_and_run_functions()
    index = index_func()
    query_results, gts = query_func(index)
    for r, gt in zip(query_results, gts):
        assert r[0] == gt


@pytest.mark.unit
def test_doc_retrieval_serialization():
    index_func, query_func = get_build_and_run_functions()
    index1 = index_func()
    index1.serialize_to_file("test.serialized")
    index2 = thirdai.search.doc_retrieval_index.deserialize_from_file("test.serialized")
    query_results1, gts1 = query_func(index1)
    query_results2, gts2 = query_func(index2)
    for r1, r2 in zip(query_results1, query_results2):
        for a, b in zip(r1, r2):
            assert a == b
