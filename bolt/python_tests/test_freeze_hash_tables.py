from thirdai import bolt
<<<<<<< HEAD:bolt/python_tests/dag_python_tests/test_bolt_dag_freeze_hash_tables.py
from ..utils import gen_numpy_training_data, get_simple_dag_model
=======
from utils import gen_numpy_training_data
>>>>>>> callback-interface:bolt/python_tests/test_freeze_hash_tables.py
import pytest

pytestmark = [pytest.mark.unit]


def test_freeze_dag_hash_tables():
    n_classes = 100

    model = get_simple_dag_model(
        input_dim=n_classes,
        hidden_layer_dim=1000,
        hidden_layer_sparsity=0.15,
        output_dim=n_classes,
    )

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

    # Freeze hash tables and train for 2 more epochs.
    model.freeze_hash_tables()

    model.train(data, labels, train_config)

    test_metrics2 = model.predict(data, labels, predict_config)[0]
    assert test_metrics2["categorical_accuracy"] >= 0.9
    assert test_metrics2["categorical_accuracy"] > test_metrics1["categorical_accuracy"]
