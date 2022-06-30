from thirdai import bolt, dataset
from utils import gen_training_data
import pytest
import numpy


@pytest.mark.unit
def test_concat_train_sparse():
    num_classes = 100
    hidden_layer_total_dim = 200
    hidden_layer_sparsity = 0.1

    input_layer = bolt.graph.Input(dim=num_classes)

    hidden_layer_1 = bolt.graph.FullyConnected(
        dim=hidden_layer_total_dim // 2,
        sparsity=hidden_layer_sparsity,
        activation="relu",
    )(input_layer)

    hidden_layer_2 = bolt.graph.FullyConnected(
        dim=hidden_layer_total_dim // 2,
        sparsity=0.1,
        activation="relu",
    )(input_layer)

    concate_layer = bolt.graph.Concatenate()([hidden_layer_1, hidden_layer_2])

    output_layer = bolt.graph.FullyConnected(dim=num_classes, activation="softmax")(
        concate_layer
    )

    model = bolt.graph.Model(inputs=[input_layer], output=output_layer)

    model.compile(loss=bolt.CategoricalCrossEntropyLoss())

    train_data, train_labels = gen_training_data(n_classes=num_classes)

    train_config = bolt.graph.TrainConfig.makeConfig(learning_rate=0.001, epochs=3)

    metrics = model.train_np(
        train_data=train_data,
        batch_size=256,
        train_labels=train_labels,
        train_config=train_config,
    )

    predict_config = bolt.graph.PredictConfig.makeConfig().withMetrics(
        ["categorical_accuracy"]
    )

    metrics = model.predict_np(
        test_data=train_data,
        batch_size=256,
        test_labels=train_labels,
        predict_config=predict_config,
    )

    assert metrics["categorical_accuracy"] >= 0.9
