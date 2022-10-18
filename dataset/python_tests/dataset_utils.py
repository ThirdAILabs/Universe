import numpy as np


def dense_vectors_to_numpy(vectors):
    return np.array([v.numpy() for v in vectors])


def sparse_vectors_to_numpy(vectors):
    indices_list = []
    values_list = []
    for vec in vectors:
        (i, v) = vec.numpy()
        indices_list.append(i)
        values_list.append(v)

    indices = np.array(indices_list)
    values = np.array(values_list)
    return (indices, values)
