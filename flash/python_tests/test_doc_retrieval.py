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

@pytest.mark.unit
def test_add_doc_find_centroids_is_fast():
  """
  The idea of this test is the time to add a doc without the centroids shouldn't
  be that much more than the time to add a doc if we find the centroids with 
  numpy and pass them in (this uses the assumption that finding the centroids is
  the slowest part). Note the query process uses the same code to find
  the closest centroids so this also will test the performance of that code.
  This could be flaky if suddenly the machine gets increased load in the middle
  of the test, but this seems unlikely.
  """
  import numpy as np
  import time
  import thirdai

  num_centroids = 2 ** 18  # 262144
  data_dim = 128
  words_per_doc = 256
  num_docs = 10
  max_percent_slowdown = 0.05
  centroids = np.random.rand(num_centroids, data_dim)
  centroids_transposed = centroids.transpose().copy()
  docs = np.random.rand(num_docs, words_per_doc, data_dim)

  doc_index_precomputed_centroids = thirdai.search.doc_retrieval_index(
    centroids=centroids,
    hashes_per_table=8,
    num_tables=8,
    dense_input_dimension=data_dim,    
  )
  doc_index_compute_centroids = thirdai.search.doc_retrieval_index(
    centroids=centroids,
    hashes_per_table=8,
    num_tables=8,
    dense_input_dimension=data_dim,    
  )

  numpy_start = time.time()
  centroid_ids_list = []
  for doc in docs:
    dot = doc.dot(centroids_transposed)
    centroid_ids_list.append(np.argmax(dot, axis=1))
  avg_numpy_time = (time.time() - numpy_start) / len(docs)

  precomputed_start = time.time()
  for i, (doc, centroid_ids) in enumerate(zip(docs, centroid_ids_list)):
    doc_index_precomputed_centroids.add_document(doc_id=str(i), doc_text="test", doc_embeddings=np.array(doc), centroid_ids=centroid_ids.flatten())
  avg_precomputed_time = (time.time() - precomputed_start) / len(docs)

  with_compute_start = time.time()
  for i, doc in enumerate(docs):
    doc_index_compute_centroids.add_document(doc_id=str(i), doc_text="test", doc_embeddings=np.array(doc))
  avg_with_compute_time = (time.time() - with_compute_start) / len(docs)


  print(avg_numpy_time, avg_precomputed_time, avg_with_compute_time)
  print((avg_with_compute_time - avg_precomputed_time) / avg_numpy_time, (1 + max_percent_slowdown))
  assert (avg_with_compute_time - avg_precomputed_time) / avg_numpy_time < (1 + max_percent_slowdown)
  
# TOOD(josh): add test with lots of elements to ensure centroid reranking is reasonably fast 

# TODO(josh): add overall general benchmarking test, perhaps download msmarco index or access saved, or similar

test_add_doc_find_centroids_is_fast()
