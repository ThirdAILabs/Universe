import numpy as np
import pytest
from thirdai import bolt_v2 as bolt

from dataset import create_dataset


INPUT_DIMS = (20, 30, 40)
BATCH_SIZE = 10
N_BATCHES = 5


@pytest.mark.unit
def test_sum_op():
    input_1 = bolt.nn.Input(dims=INPUT_DIMS)
    input_2 = bolt.nn.Input(dims=INPUT_DIMS)

    sum_layer = bolt.nn.Sum()(input_1, input_2)

    loss = bolt.nn.losses.CategoricalCrossEntropy(
        sum_layer, bolt.nn.Input(dims=INPUT_DIMS)
    )

    model = bolt.nn.Model(inputs=[input_1, input_2], outputs=[sum_layer], losses=[loss])

    data_x_1, data_y = create_dataset(
        shape=(BATCH_SIZE, *INPUT_DIMS), n_batches=N_BATCHES, with_grad=True
    )
    data_x_2, _ = create_dataset(
        shape=(BATCH_SIZE, *INPUT_DIMS), n_batches=N_BATCHES, with_grad=True
    )

    for x_1, x_2, y in zip(data_x_1, data_x_2, data_y):
        model.train_on_batch(x_1 + x_2, y)

        output = sum_layer.tensor().activations
        expected_output = x_1[0].activations + x_2[0].activations

        assert np.array_equal(output, expected_output)
