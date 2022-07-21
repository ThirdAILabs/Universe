from thirdai import dataset
import numpy as np


def generate_dense_bolt_dataset_from_numpy(rows, cols):
    test_numpy = np.tile(np.arange(0, cols), (rows, 1)).astype("float32")
    return dataset.from_numpy(test_numpy, batch_size=64)


def check_dense_bolt_dataset(np_dataset, cols):
    for batch_index in range(len(np_dataset)):
        batch = np_dataset[batch_index]
        for row_index in range(len(batch)):
            assert str(batch[row_index]) == str(list(range(cols)))


def test_dense_numpy():
    num_arrays = 100
    datasets = [
        generate_dense_bolt_dataset_from_numpy(rows=10, cols=20)
        for _ in range(num_arrays)
    ]
    for np_dataset in datasets:
        check_dense_bolt_dataset(np_dataset, cols=20)


def test_sparse_numpy():
    pass


def test_bad_numpy():
    pass


def test_tokens_numpy():
    pass
