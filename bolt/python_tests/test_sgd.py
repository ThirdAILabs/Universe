import pytest
from thirdai import bolt

from utils import gen_numpy_training_data, get_simple_dag_model


@pytest.mark.unit
def test_sgd():
    n_classes = 20
    train_data, train_labels = gen_numpy_training_data(
        n_classes=n_classes, n_samples=5000, batch_size_for_conversion=100
    )
    test_data, test_labels = gen_numpy_training_data(
        n_classes=n_classes, n_samples=1000
    )

    model = get_simple_dag_model(
        input_dim=n_classes,
        hidden_layer_dim=100,
        hidden_layer_sparsity=1.0,
        output_dim=n_classes,
        optimizer=bolt.nn.optimizers.Sgd(),
    )

    predict_config = bolt.graph.PredictConfig.make().with_metrics(
        ["categorical_accuracy"]
    )
    train_config = bolt.graph.TrainConfig.make(epochs=2, learning_rate=1.0)

    model.train(train_data, train_labels, train_config)
    metrics = model.predict(test_data, test_labels, predict_config)[0]

    assert metrics["categorical_accuracy"] >= 0.9
