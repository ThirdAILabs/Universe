from thirdai import bolt
from ..utils import gen_training_data
import pytest

pytestmark = [pytest.mark.unit]


def test_freeze_dag_hash_tables():
    # Define and compile model.
    n_classes = 100
    input_layer = bolt.graph.Input(dim=n_classes)
    hidden_layer = bolt.graph.FullyConnected(
        dim=1000, sparsity=0.15, activation="relu"
    )(input_layer)
    output_layer = bolt.graph.FullyConnected(dim=n_classes, activation="softmax")(
        hidden_layer
    )

    model = bolt.graph.Model(inputs=[input_layer], output=output_layer)
    model.compile(bolt.CategoricalCrossEntropyLoss())

    # Generate dataset.
    data, labels = gen_training_data(n_classes=n_classes, n_samples=10000)

    # Train and predict before freezing hash tables.
    train_config = bolt.graph.TrainConfig.make(
        learning_rate=0.001, epochs=2
    ).with_batch_size(100)
    model.train(data, labels, train_config)

    predict_config = (
        bolt.graph.PredictConfig.make()
        .enable_sparse_inference()
        .with_metrics(["categorical_accuracy"])
    )
    test_metrics1 = model.predict(data, labels, predict_config)[0]

    assert test_metrics1["categorical_accuracy"] >= 0.8

    # Freeze hash tables and train for 2 more epochs.
    model.freeze_hash_tables()

    model.train(data, labels, train_config)

    test_metrics2 = model.predict(data, labels, predict_config)[0]
    assert test_metrics2["categorical_accuracy"] >= 0.9
    assert test_metrics2["categorical_accuracy"] > test_metrics1["categorical_accuracy"]
