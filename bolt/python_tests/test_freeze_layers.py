import pytest
from thirdai import bolt

from utils import gen_numpy_training_data

pytestmark = [pytest.mark.unit]


def test_freeze_dag_hash_tables():
    n_classes = 100

    input_dim = n_classes
    hidden_layer_dim = 1000
    hidden_layer_sparsity = 0.15
    output_dim = n_classes
    output_activation = "softmax"
    loss = (bolt.CategoricalCrossEntropyLoss(),)

    input_layer = bolt.graph.Input(dim=input_dim)

    hidden_layer = bolt.graph.FullyConnected(
        dim=hidden_layer_dim, sparsity=hidden_layer_sparsity, activation="relu"
    )(input_layer)

    output_layer = bolt.graph.FullyConnected(
        dim=output_dim, activation=output_activation
    )(hidden_layer)

    model = bolt.graph.Model(inputs=[input_layer], output=output_layer)

    model.compile(loss)

    # Generate dataset.
    data, labels = gen_numpy_training_data(n_classes=n_classes, n_samples=10000)

    # Train and predict before freezing hash tables.
    train_config = bolt.graph.TrainConfig.make(learning_rate=0.001, epochs=2)
    model.train(data, labels, train_config)

    predict_config = (
        bolt.graph.PredictConfig.make()
        .enable_sparse_inference()
        .with_metrics(["categorical_accuracy"])
    )

    test_metrics1 = model.predict(data, labels, predict_config)[0]
    assert test_metrics1["categorical_accuracy"] >= 0.8

    hidden_layer.trainable(False)
    # weights = hidden_layer.get_weights()

    output_layer_new = bolt.graph.FullyConnected(
        dim=output_dim, activation=output_activation
    )(hidden_layer)

    model_new = bolt.graph.Model(inputs=[input_layer], output=output_layer_new)
    model.compile(loss)
    model_new.train(data, labels, train_config)
    test_metrics2 = model_new.predict(data, labels, predict_config)[0]
    assert test_metrics2["categorical_accuracy"] >= 0.8
