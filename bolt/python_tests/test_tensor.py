import numpy as np
import pytest
from thirdai import bolt


@pytest.mark.unit
def test_dense_tensor_to_numpy():
    arr = np.arange(120, dtype=np.float32).reshape((6, 20))

    tensor = bolt.nn.Tensor(arr)

    assert tensor.active_neurons == None

    assert np.array_equal(arr, tensor.activations)

    assert tensor.gradients == None


@pytest.mark.unit
def test_sparse_tensor_to_numpy():
    dense_dim = 10000
    indices = np.random.randint(0, dense_dim, size=(6, 20), dtype=np.uint32)

    values = np.arange(120, dtype=np.float32).reshape((6, 20))

    tensor = bolt.nn.Tensor(indices, values, dense_dim=dense_dim)

    assert np.array_equal(indices, tensor.active_neurons)

    assert np.array_equal(values, tensor.activations)

    assert tensor.gradients == None
