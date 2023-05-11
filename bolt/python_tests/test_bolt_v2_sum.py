import numpy as np
import pytest
from thirdai import bolt_v2 as bolt


@pytest.mark.unit
@pytest.mark.parametrize("sparsity", [0.4, 1.0])
def test_tanh_op(sparsity):
    batch_size = 10
    seq_len = 20
    dim = 100

    input_layer = bolt.nn.Input(dims=(seq_len, dim))

    hidden_layer = bolt.nn.FullyConnected(
        dim=dim, input_dim=dim, sparsity=sparsity, activation="linear"
    )(input_layer)

    output_layer = bolt.nn.Sum()(hidden_layer)

    labels = bolt.nn.Input(dim=dim)

    loss = bolt.nn.losses.CategoricalCrossEntropy(
        activations=output_layer, labels=labels
    )

    model = bolt.nn.Model(inputs=[input_layer], outputs=[output_layer], losses=[loss])

    for _ in range(4):
        inputs = np.random.uniform(0.0, 10.0, size=(batch_size, seq_len, dim))
        x = bolt.nn.Tensor(inputs)

        labels = np.random.randint(0, dim, size=(batch_size, 1), dtype=np.uint32)
        y = bolt.nn.Tensor(indices=labels, values=np.ones_like(labels), dense_dim=dim)

        model.train_on_batch([x], [y])
        model.update_parameters(0.0001)

        assert output_layer.tensor().indices == None

        if sparsity < 1.0:
            indices = hidden_layer.tensor().indices
            values = hidden_layer.tensor().values
            output_grads = output_layer.tensor().gradients

            dense_sum = np.zeros((batch_size, dim), dtype=np.float32)

            hidden_grads = np.zeros_like(indices, dtype=np.float32)

            for i in range(indices.shape[0]):
                for j in range(indices.shape[1]):
                    dense_vec = np.zeros(dim)
                    dense_vec[indices[i, j]] = values[i, j]
                    dense_sum[i] += dense_vec

                    hidden_grads[i, j] = output_grads[i, indices[i, j]]

            assert np.allclose(output_layer.tensor().values, dense_sum)

            assert np.allclose(hidden_grads, hidden_layer.tensor().gradients)
        else:
            assert np.array_equal(
                output_layer.tensor().values,
                np.sum(hidden_layer.tensor().values, axis=1),
            )

            # Outputs have shape (batch_size, input_dim) after the sum along the
            # sequence length. This reshapes it to (batch_size, seq_len, input_dim)
            # by copying the vectors in the last dimension seq_len times.
            assert np.array_equal(
                hidden_layer.tensor().gradients,
                np.broadcast_to(
                    np.expand_dims(output_layer.tensor().gradients, 1),
                    (batch_size, seq_len, dim),
                ),
            )
