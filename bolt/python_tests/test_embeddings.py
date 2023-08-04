import numpy as np
import pytest
from thirdai import bolt, dataset


def get_embedding_model(input_dim, embedding_factory):
    input_1 = bolt.nn.Input(dim=input_dim)

    input_2 = bolt.nn.Input(dim=input_dim)

    robez_bottom = embedding_factory()(input_1)

    robez_top = embedding_factory()(input_2)

    concat_layer = bolt.nn.Concatenate()([robez_bottom, robez_top])

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
    data = bolt.train.convert_datasets(
        datasets=[
            dataset.from_numpy(data_1, batch_size=batch_size),
            dataset.from_numpy(data_2, batch_size=batch_size),
        ],
        dims=[input_dim, input_dim],
    )

    return data, labels, chunk(labels_np, batch_size)


INPUT_DIM = 10
NUM_TRAIN = 10000
NUM_TEST = 100


def train_and_evaluate_embedding_model(embedding_factory):
    # This test ensures we can learn how to add (almost certainly by memorizing)
    # The real thing it tests is 1. multiple inputs and 2. numpy to token datasets

    model = get_embedding_model(INPUT_DIM, embedding_factory)

    train_data, train_labels, _ = generate_sum_datasets_and_labels(
        input_dim=INPUT_DIM, num_examples=NUM_TRAIN
    )

    for _ in range(5):
        for x, y in zip(train_data, train_labels):
            model.train_on_batch(x, y)
            model.update_parameters(0.01)

    test_data, _, test_labels_np = generate_sum_datasets_and_labels(
        input_dim=INPUT_DIM,
        num_examples=NUM_TEST,
    )

    correct = 0
    total = 0
    for x, y_np in zip(test_data, test_labels_np):
        output = model.forward(x, use_sparsity=False)

        correct += np.sum(np.argmax(output[0].activations, axis=1) == y_np)
        total += len(y_np)

    assert correct / total > 0.8


@pytest.mark.unit
def test_robez_op():
    def robez_factory():
        return bolt.nn.RobeZ(
            num_embedding_lookups=4,
            lookup_size=8,
            log_embedding_block_size=10,
            reduction="sum",
        )

    train_and_evaluate_embedding_model(robez_factory)


@pytest.mark.unit
def test_embedding_op():
    def embedding_factory():
        return bolt.nn.Embedding(dim=32, input_dim=INPUT_DIM, activation="linear")

    train_and_evaluate_embedding_model(embedding_factory)
