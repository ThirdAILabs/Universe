import thirdai
import pytest
import numpy as np
import random

@pytest.mark.unit
def test_near_duplicate_numpy():

  data_dim = 100
  dataset_size = 100000
  queries_size = 1000
  dataset_std = 1
  queries_std = 0.1

  # Generate n points using gaussian
  np.random.seed(42)
  random.seed(42)
  dataset = np.random.normal(size=(dataset_size, data_dim), scale=dataset_std)

  # Generate queries from random points
  queries = []
  gts = []
  for i in range(queries_size):
    gt = random.randrange(dataset_size)
    query = dataset[gt] + np.random.normal(size=(data_dim), scale=queries_std)
    queries.append(query)
    gts.append(gt)
  queries = np.array(queries)

  # Add to mag search
  hf = thirdai.hashing.SignedRandomProjection(input_dim=data_dim, hashes_per_table=16, num_tables=20)
  index = thirdai.search.MagSearch(hf, reservoir_size=100)
  
  index.add(dense_data=dataset, starting_index=0)
  results = index.query(queries, top_k=1)
  recall = sum([gt == result[0] for gt, result in zip(gts, results)]) / queries_size
  assert recall == 1

@pytest.mark.unit
def test_exact_duplicate_sparse():

  data_dim = 1000000
  min_num_non_zero = 5
  max_num_non_zero = 20
  dataset_size = 10000

  data_values = []
  data_indices = []

  random.seed(42)
  np.random.seed(42)
  for i in range(dataset_size):
    num_non_zero = random.randrange(min_num_non_zero, max_num_non_zero)
    
    data_values.append(np.random.normal(size=(num_non_zero,)))

    indices = set()
    while len(indices) < num_non_zero:
      indices.add(random.randrange(0, data_dim))
    data_indices.append(np.asarray(sorted(list(indices))))

  # Add to mag search
  hf = thirdai.hashing.MinHash(hashes_per_table=10, num_tables=20, range=100000)
  index = thirdai.search.MagSearch(hf, reservoir_size=10000)
  
  index.add(sparse_values=data_values, sparse_indices=data_indices, starting_index=0)
  results = index.query(sparse_query_values=data_values, sparse_query_indices=data_indices, top_k=5)
  for i in range(len(results)):
    assert results[i][0] == i

test_exact_duplicate_sparse()