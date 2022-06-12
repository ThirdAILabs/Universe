import numpy as np
from thirdai.search import BoltSearch
from tqdm import tqdm
import os


def get_gt(glove_data, train_data, top_k):
    filename = f"glove_train_topk_{top_k}_{len(train_data)}_{len(glove_data)}.npy"
    if os.path.exists(filename):
        return np.load(filename)
    all_top_indices = []
    all_data_normed = glove_data / np.linalg.norm(glove_data, axis=1, keepdims=True)
    for v in tqdm(list(range(len(train_data)))):
        training_normed = train_data[v : v + 1] / np.linalg.norm(
            train_data[v : v + 1], axis=1, keepdims=True
        )
        dot = np.matmul(training_normed, all_data_normed.transpose())[0]
        top_indices = np.argpartition(dot, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(dot[top_indices])][::-1]
        all_top_indices.append(top_indices.copy())
    all_top_indices = np.array(all_top_indices)
    np.save(filename, all_top_indices)
    return all_top_indices


def test_glove():
    glove_data = np.load("glove_data.npy")
    glove_queries = np.load("glove_queries.npy")
    glove_neighbors = np.load("glove_neighbors.npy")

    train_data = glove_data[:100000]
    train_gt = get_gt(glove_data, train_data, top_k=100)

    search_index = BoltSearch(
        estimated_dataset_size=10**6, num_classifiers=4, input_dim=100
    )

    # Real example:
    search_index.index(
        train_data=train_data, train_gt=train_gt, all_data=glove_data, batch_size=2048
    )

    # Small example, weird that it performs badly:
    # search_index.index(
    #     train_data=train_data[:100],
    #     train_gt=train_gt,
    #     all_data=glove_data,
    #     batch_size=2048,
    #     num_epochs_per_iteration=100,
    #     num_iterations=10,
    #     learning_rate=0.01,
    #     num_alternative_groups_to_consider=5,
    #     num_label_neighbors=1
    # )


test_glove()
