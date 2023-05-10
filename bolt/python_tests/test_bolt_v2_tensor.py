import numpy as np
import pytest
from thirdai import bolt_v2 as bolt


@pytest.mark.unit
def test_dense_tensor_to_numpy():
    arr = np.arange(120, dtype=np.float32).reshape((5, 4, 3, 2))

    tensor = bolt.nn.Tensor(arr)

    assert tensor.indices == None

    assert np.array_equal(arr, tensor.values)

    assert tensor.gradients == None


@pytest.mark.unit
def test_sparse_tensor_to_numpy():
    dense_dim = 10000
    indices = np.random.randint(0, dense_dim, size=(5, 4, 3, 2), dtype=np.uint32)

    values = np.arange(120, dtype=np.float32).reshape((5, 4, 3, 2))

    tensor = bolt.nn.Tensor(indices, values, dense_dim=dense_dim)

    assert np.array_equal(indices, tensor.indices)

    assert np.array_equal(values, tensor.values)

    assert tensor.gradients == None


@pytest.mark.unit
def test_sparse_tensor_to_numpy_no_values():
    dense_dim = 10000
    indices = np.random.randint(0, dense_dim, size=(5, 4, 3, 2), dtype=np.uint32)

    tensor = bolt.nn.Tensor(indices, values=None, dense_dim=dense_dim)

    assert np.array_equal(indices, tensor.active_neurons)

    assert np.array_equal(np.ones((5, 4, 3, 2), dtype=np.float32), tensor.activations)

    assert tensor.gradients == None
