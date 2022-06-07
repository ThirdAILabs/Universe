import numpy as np
from thirdai.search import BoltSearch


def test_glove():
    glove_data = np.load("glove_data.npy")
    glove_queries = np.load("glove_queries.npy")
    glove_neighbors = np.load("glove_neighbors.npy")

    search_index = BoltSearch(
        estimated_dataset_size=10**6, num_classifiers=5, input_dim=100
    )
    search_index.index(glove_data)
