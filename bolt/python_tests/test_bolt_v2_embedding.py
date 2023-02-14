import numpy as np
import pytest
from thirdai import bolt_v2 as bolt
from thirdai import dataset


def get_sum_model(input_dim):

    input_1 = bolt.nn.Input(dim=input_dim)

    input_2 = bolt.nn.Input(dim=input_dim)

    embedding_bottom = bolt.nn.Embedding(
        num_embedding_lookups=4,
        lookup_size=8,
        log_embedding_block_size=10,
        update_chunk_size=8,
        reduction="sum",
    )(input_1)

    embedding_top = bolt.nn.Embedding(
        num_embedding_lookups=4,
        lookup_size=8,
        log_embedding_block_size=10,
        update_chunk_size=8,
        reduction="sum",
    )(input_2)

    concat_layer = bolt.nn.Concatenate()([embedding_bottom, embedding_top])

    output_layer = bolt.nn.FullyConnected(
        dim=input_dim * 2, input_dim=concat_layer.dim(), activation="softmax"
    )(concat_layer)

    labels = bolt.nn.Input(dim=input_dim * 2)

    loss = bolt.nn.losses.CategoricalCrossEntropy(
        activations=output_layer, labels=labels
    )

    model = bolt.nn.Model(
        inputs=[input_1, input_2], outputs=[output_layer], losses=[loss]
    )

    return model


def chunk(array, chunksize):
    chunks = []
    for i in range(0, len(array), chunksize):
        chunks.append(np.array(array[i : i + chunksize]))
    return chunks


def generate_sum_datasets_and_labels(input_dim, num_examples, batch_size=64):
    data_1 = np.random.randint(0, input_dim, size=num_examples, dtype="uint32")
    data_2 = np.random.randint(0, input_dim, size=num_examples, dtype="uint32")

    labels_np = data_1 + data_2

    labels = bolt.train.convert_dataset(
        dataset.from_numpy(labels_np, batch_size=batch_size), dim=input_dim * 2
    )
    data_1 = bolt.train.convert_dataset(
        dataset.from_numpy(data_1, batch_size=batch_size), dim=input_dim
    )
    data_2 = bolt.train.convert_dataset(
        dataset.from_numpy(data_2, batch_size=batch_size), dim=input_dim
    )
    return data_1, data_2, labels, chunk(labels_np, batch_size)


# This test ensures we can learn how to add (almost certainly by memorizing)
# The real thing it tests is 1. multiple inputs and 2. numpy to token datasets
@pytest.mark.unit
def test_embedding_op():

    input_dim = 10
    num_train = 10000
    num_test = 100

    model = get_sum_model(input_dim)

    train_1, train_2, train_labels, _ = generate_sum_datasets_and_labels(
        input_dim=input_dim, num_examples=num_train
    )

    for _ in range(5):
        for x1, x2, y in zip(train_1, train_2, train_labels):
            model.train_on_batch([x1, x2], [y])
            model.update_parameters(0.01)

    test_1, test_2, _, test_labels = generate_sum_datasets_and_labels(
        input_dim=input_dim,
        num_examples=num_test,
    )

    correct = 0
    total = 0
    for x1, x2, y in zip(test_1, test_2, test_labels):
        output = model.forward([x1, x2], use_sparsity=False)

        correct += np.sum(np.argmax(output[0].activations, axis=1) == y)
        total += len(y)

    assert correct / total > 0.8
