import pytest

pytestmark = [pytest.mark.unit, pytest.mark.release]

import numpy as np
from thirdai import bolt

from utils import (
    gen_single_sparse_layer_network,
    gen_numpy_training_data,
    get_categorical_acc,
    train_network,
    build_simple_distributed_bolt_network,
    train_single_node_distributed_network,
)


# gets compressed gradients as (indices,values) tuple and then asserts that the gradient matrix after setting is the same as the tuple
def test_dragon_compression():
    network = build_simple_distributed_bolt_network(sparsity=1.0, n_classes=10)
    examples, labels = gen_numpy_training_data(n_classes=10, n_samples=1000)
    train_single_node_distributed_network(
        network, examples, labels, epochs=1, update_parameters=False
    )

    indices, values = network.get_indexed_sketch_for_gradients(
        layer_index=0, compression_density=0.1, sketch_biases=False, seed_for_hashing=0
    )
    network.set_gradients_from_indices_values(
        layer_index=0, indices=indices, values=values, set_biases=False
    )

    network_weights = network.get_weights_gradients(0).flatten()
    np.add.at(network_weights, indices, -1 * values)
    norm_after_subtracting_gradients = np.linalg.norm(network_weights)

    assert norm_after_subtracting_gradients == 0
