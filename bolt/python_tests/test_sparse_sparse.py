import numpy as np
import pytest

from utils import build_train_and_predict_single_hidden_layer

pytestmark = [pytest.mark.unit]


def run_sparse_sparse(optimize_sparse_sparse, accuracy_threshold):
    input_output_dim = 100
    input_num_nonzeros = 1
    num_examples = 10000
    values_np = np.ones(input_num_nonzeros * num_examples).astype("float32")
    indices_np = np.random.randint(
        input_output_dim, size=input_num_nonzeros * num_examples
    ).astype("uint32")
    offsets_np = np.arange(
        0, (num_examples + 1) * input_num_nonzeros, input_num_nonzeros
    ).astype("uint32")
    labels_np = np.reshape(indices_np, (-1, 1))
    res = build_train_and_predict_single_hidden_layer(
        (indices_np, values_np, offsets_np),
        labels_np,
        input_output_dim,
        output_sparsity=0.1,
        optimize_sparse_sparse=optimize_sparse_sparse,
    )
    print(res[0]["categorical_accuracy"])
    assert res[0]["categorical_accuracy"] > accuracy_threshold


# Goal is to learn the identity function
def test_sparse_sparse_optimization_enabled():
    run_sparse_sparse(optimize_sparse_sparse=True, accuracy_threshold=0.2)


def test_sparse_sparse_optimization_disabled():
    run_sparse_sparse(optimize_sparse_sparse=False, accuracy_threshold=0.2)
