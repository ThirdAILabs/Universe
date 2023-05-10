import numpy as np
import pytest
from thirdai import bolt_v2 as bolt
from thirdai import dataset


def get_sum_model(samples_per_batch, input_dim):
    sample_shape = samples_per_batch[1:]  # Discard batch_size
    input_1 = bolt.nn.Input(dims=(*sample_shape, input_dim))

    input_2 = bolt.nn.Input(dims=(*sample_shape, input_dim))

    embedding_bottom = bolt.nn.Embedding(
        num_embedding_lookups=4,
        lookup_size=8,
        log_embedding_block_size=10,
        reduction="sum",
    )(input_1)

    embedding_top = bolt.nn.Embedding(
        num_embedding_lookups=4,
        lookup_size=8,
        log_embedding_block_size=10,
        reduction="sum",
    )(input_2)

    concat_layer = bolt.nn.Concatenate()([embedding_bottom, embedding_top])

    output_layer = bolt.nn.FullyConnected(
        dim=input_dim * 2, input_dim=concat_layer.dims()[-1], activation="softmax"
    )(concat_layer)

    labels = bolt.nn.Input(dims=(*sample_shape, 2 * input_dim))

    loss = bolt.nn.losses.CategoricalCrossEntropy(
        activations=output_layer, labels=labels
    )

    model = bolt.nn.Model(
        inputs=[input_1, input_2], outputs=[output_layer], losses=[loss]
    )

    return model


def generate_sum_datasets_and_labels(input_dim, shape, n_batches):
    data_1 = np.random.randint(
        0, input_dim, size=(n_batches, *shape, 1), dtype="uint32"
    )
    data_2 = np.random.randint(
        0, input_dim, size=(n_batches, *shape, 1), dtype="uint32"
    )

    labels = data_1 + data_2

    data_batches = []
    label_batches = []

    for i in range(len(labels)):
        data_1_tensor = bolt.nn.Tensor(
            data_1[i], np.ones_like(data_1[i], dtype=np.float32), dense_dim=input_dim
        )

        data_2_tensor = bolt.nn.Tensor(
            data_2[i], np.ones_like(data_2[i], dtype=np.float32), dense_dim=input_dim
        )

        label_tensor = bolt.nn.Tensor(
            labels[i],
            np.ones_like(labels[i], dtype=np.float32),
            dense_dim=2 * input_dim,
        )

        data_batches.append([data_1_tensor, data_2_tensor])
        label_batches.append([label_tensor])

    return data_batches, label_batches, labels.reshape(labels.shape[:-1])


# This test ensures we can learn how to add (almost certainly by memorizing)
# The real thing it tests is 1. multiple inputs and 2. numpy to token datasets
@pytest.mark.unit
def test_embedding_op():
    input_dim = 10
    samples_per_batch = (8, 4, 3)

    model = get_sum_model(samples_per_batch, input_dim)

    train_data, train_labels, _ = generate_sum_datasets_and_labels(
        input_dim=input_dim, shape=samples_per_batch, n_batches=100
    )

    for _ in range(5):
        for x, y in zip(train_data, train_labels):
            model.train_on_batch(x, y)
            model.update_parameters(0.01)

    test_data, _, test_labels_np = generate_sum_datasets_and_labels(
        input_dim=input_dim, shape=samples_per_batch, n_batches=10
    )

    correct = 0
    total = 0
    for x, y_np in zip(test_data, test_labels_np):
        output = model.forward(x, use_sparsity=False)

        correct += np.sum(np.argmax(output[0].values, axis=-1) == y_np)
        total += len(y_np)

    assert correct / total > 0.8
