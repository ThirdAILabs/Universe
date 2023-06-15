import os

import h5py
import numpy as np
import pytest
from thirdai import search


@pytest.fixture(scope="session")
def download_mnist_ann_benchmark():
    url = "http://ann-benchmarks.com/mnist-784-euclidean.hdf5"

    dataset_path = "./mnist-784-euclidean.hdf5"
    if not os.path.exists(dataset_path):
        os.system(f"wget {url} -O {dataset_path}")

    data = h5py.File(dataset_path, "r")

    train = data.get("train")[()]
    test = data.get("test")[()]
    gtruth = data.get("neighbors")[()]

    return train, test, gtruth


def test_hnsw(download_mnist_ann_benchmark):
    train, test, gtruth = download_mnist_ann_benchmark

    index = search.HNSW(max_nbrs=16, data=train, construction_buffer_size=32)

    k = 100
    recall = 0.0
    for row, actual in zip(test, gtruth):
        nbrs = np.array(index.query(query=row, k=k, search_buffer_size=128))
        recall += len(np.intersect1d(nbrs, actual[:k])) / k

    avg_recall = recall / len(test)

    print(avg_recall)
    assert avg_recall >= 0.95
