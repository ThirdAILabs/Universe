import pytest
from thirdai import bolt, dataset
import numpy as np


def get_sum_model(input_dim):

    input_1 = bolt.graph.TokenInput()

    input_2 = bolt.graph.TokenInput()

    embedding_bottom = bolt.graph.Embedding(
        num_embedding_lookups=input_dim, lookup_size=16, log_embedding_block_size=10
    )(input_1)

    embedding_top = bolt.graph.Embedding(
        num_embedding_lookups=input_dim, lookup_size=16, log_embedding_block_size=10
    )(input_2)

    concat_layer = bolt.graph.Concatenate()([embedding_bottom, embedding_top])

    output_layer = bolt.graph.FullyConnected(dim=input_dim * 2, activation="softmax")(
        concat_layer
    )

    model = bolt.graph.Model(
        token_inputs=[input_1, input_2], inputs=[], output=output_layer
    )

    model.compile(loss=bolt.CategoricalCrossEntropyLoss())

    return model


def generate_sum_datasets_and_labels(input_dim, num_examples):
    data_1 = np.random.randint(0, input_dim, size=num_examples, dtype="uint32")
    data_2 = np.random.randint(0, input_dim, size=num_examples, dtype="uint32")
    labels = dataset.from_numpy(data_1 + data_2, batch_size=64)
    data_1 = dataset.tokens_from_numpy(data_1, batch_size=64)
    data_2 = dataset.tokens_from_numpy(data_2, batch_size=64)
    return data_1, data_2, labels


# This test ensures we can learn how to add (probably by memorizing)
# The real thing it tests is 1. multiple inputs and 2. numpy to token datasets
@pytest.mark.unit
def test_token_sum():

    input_dim = 10
    num_train = 10000
    num_test = 100
    num_epochs = 5

    model = get_sum_model(input_dim)

    train_1, train_2, train_labels = generate_sum_datasets_and_labels(
        input_dim=input_dim, num_examples=num_train
    )
    train_config = bolt.graph.TrainConfig.make(
        learning_rate=0.01, epochs=num_epochs
    ).silence()
    model.train(
        train_data=[],
        train_tokens=[train_1, train_2],
        train_labels=train_labels,
        train_config=train_config,
    )

    test_1, test_2, test_labels = generate_sum_datasets_and_labels(
        input_dim=input_dim, num_examples=num_test
    )
    predict_config = (
        bolt.graph.PredictConfig.make().silence().with_metrics(["categorical_accuracy"])
    )
    metrics = model.predict(
        test_data=[],
        test_tokens=[test_1, test_2],
        test_labels=test_labels,
        predict_config=predict_config,
    )
    assert metrics[0]["categorical_accuracy"] > 0.8
