import numpy as np


def dense_vectors_to_numpy(vectors):
    return np.array([v.to_numpy() for v in vectors])


def sparse_vectors_to_numpy(vectors):
    indices_list = []
    values_list = []
    for vec in vectors:
        (i, v) = vec.to_numpy()
        indices_list.append(i)
        values_list.append(v)

    indices = np.array(indices_list)
    values = np.array(values_list)
    return (indices, values)


def get_bolt_vectors_from_dataset(dataset):
    vectors = []
    for batch in range(len(dataset)):
        for vec in range(len(dataset[batch])):
            vectors.append(dataset[batch][vec])
    return vectors


def dense_bolt_dataset_to_numpy(dataset):
    return dense_vectors_to_numpy(get_bolt_vectors_from_dataset(dataset))


def sparse_bolt_dataset_to_numpy(dataset):
    return sparse_vectors_to_numpy(get_bolt_vectors_from_dataset(dataset))
