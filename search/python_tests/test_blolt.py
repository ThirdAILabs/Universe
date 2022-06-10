import numpy as np
from thirdai.search import BoltSearch


def test_glove():
    glove_data = np.load("glove_data.npy")
    glove_queries = np.load("glove_queries.npy")
    glove_neighbors = np.load("glove_neighbors.npy")

    # # # Get 70% accuracy with num_groups_to_consider=5, basically groups all exactly the same size
    # # # num_groups_to_consider=3 sucks, and kinda so does 4. 6 gives good groups, but not as high accuracy.
    # search_index = BoltSearch(
    #     estimated_dataset_size=10**6, num_classifiers=2, input_dim=100
    # )
    # search_index.index(
    #     train_data=glove_data[:10000], all_data=glove_data, batch_size=2048
    # )

    # search_index.serialize_to_file("temp.serialized")

    search_index = BoltSearch.deserialize_from_file("temp.serialized")

    results = search_index.query(glove_data[:100], top_k=100)
    for i, r in enumerate(results):
        assert i in r

    results = search_index.query(glove_neighbors, top_k=100)
    recalled_in_100 = 0
    for found, gt in zip(results, glove_neighbors):
        if gt[0] in found:
            recalled_in_100 += 1
    print(recalled_in_100, flush=True)
