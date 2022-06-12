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

    # from thirdai import bolt
    # layers = [
    #     bolt.FullyConnected(
    #         dim=1024,
    #         sparsity=1,
    #         activation_function=bolt.ActivationFunctions.ReLU,
    #     ),
    #     bolt.FullyConnected(
    #         dim=10000, activation_function=bolt.ActivationFunctions.Softmax, sparsity=0.01
    #     ),
    # ]
    # network = bolt.Network(layers=layers, input_dim=100)

    # train_data_size = 1000
    # train_labels=(np.random.randint(0, 10000, size=(train_data_size,)), np.ones(shape=(train_data_size,)), np.array(range(train_data_size + 1))) 
    # network.train(train_data=train_data[:train_data_size], 
    #               train_labels=train_labels,
    #               batch_size=1024,
    #               loss_fn=bolt.CategoricalCrossEntropyLoss(),
    #               learning_rate=0.01,
    #               epochs=1,
    #               verbose=True
    #     )
    # network.predict(
    #     test_data=train_data[:train_data_size],
    #     test_labels=train_labels,
    #     batch_size=1024,
    #     metrics=["categorical_accuracy"],
    #     verbose=True,
    # )


    search_index = BoltSearch(
        estimated_dataset_size=10**6, num_classifiers=2, input_dim=100
    )
    search_index.index(
        train_data=train_data, train_gt=train_gt, all_data=glove_data, batch_size=2048
    )

    search_index.serialize_to_file("temp.serialized")

    search_index = BoltSearch.deserialize_from_file("temp.serialized")

    # print(glove_data[3])
    # print(glove_data[4])
    results = search_index.query(glove_data[3:5], top_k=10)
    print(results)
    # for i, r in enumerate(results):
    #     assert i in r

    # results = search_index.query(glove_neighbors, top_k=100)
    # recalled_in_100 = 0
    # for i, (found, gt) in enumerate(list(zip(results, glove_neighbors))):
    #     if gt[0] in found:
    #         recalled_in_100 += 1
    #         print(found, gt)
    # print(recalled_in_100, flush=True)

test_glove()