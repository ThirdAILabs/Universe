import numpy as np
from thirdai.search import BoltSearch


def test_glove():
    glove_data = np.load("glove_data.npy")
    glove_queries = np.load("glove_queries.npy")
    glove_neighbors = np.load("glove_neighbors.npy")

    # Get 70% accuracy with num_groups_to_consider=5, basically groups all exactly the same size
    # num_groups_to_consider=3 sucks, and kinda so does 4. 6 gives good groups, but not as high accuracy.
    search_index = BoltSearch(
        estimated_dataset_size=10**6,
        num_classifiers=3,
        input_dim=100,
        num_groups_to_consider=5,
    )
    search_index.index(glove_data)
