from cgi import test
from thirdai import bolt
from ..utils import gen_numpy_training_data
import pytest

pytestmark = [pytest.mark.unit]

N_CLASSES = 10


def create_simple_dag_model():
    input_layer = bolt.graph.Input(dim=N_CLASSES)
    hidden_layer = bolt.graph.FullyConnected(dim=1000, activation="relu")(input_layer)
    output_layer = bolt.graph.FullyConnected(dim=N_CLASSES, activation="softmax")(
        hidden_layer
    )

    model = bolt.graph.Model(inputs=[input_layer], output=output_layer)
    model.compile(bolt.CategoricalCrossEntropyLoss())

    return model


def train_overfitted_model(train_data, train_labels):
    model = create_simple_dag_model()

    train_config = bolt.graph.TrainConfig.make(
        learning_rate=0.001, epochs=20
    ).with_metrics(["categorical_accuracy"])

    model.train(train_data, train_labels, train_config)

    return model


def train_early_stop_model(train_data, train_labels, valid_data, valid_labels):
    model = create_simple_dag_model()

    predict_config = (
        bolt.graph.PredictConfig.make()
        .with_metrics(["categorical_accuracy"])
        .enable_sparse_inference()
    )

    train_config = (
        bolt.graph.TrainConfig.make(learning_rate=0.001, epochs=20)
        .with_metrics(["categorical_accuracy"])
        .with_early_stop_validation(
            valid_data=valid_data, valid_labels=valid_labels, patience=3, predict_config=predict_config
        )
    )

    model.train(train_data, train_labels, train_config)

    return model


def test_early_stop_validation():
    train_data, train_labels = gen_numpy_training_data(
        n_classes=N_CLASSES, n_samples=100, noise_std=0.3
    )
    valid_data, valid_labels = gen_numpy_training_data(
        n_classes=N_CLASSES, n_samples=1000
    )
    test_data, test_labels = gen_numpy_training_data(
        n_classes=N_CLASSES, n_samples=1000
    )

    overfitted_model = train_overfitted_model(train_data, train_labels)
    early_stop_model = train_early_stop_model(
        train_data, train_labels, valid_data, valid_labels
    )

    predict_config = (
        bolt.graph.PredictConfig.make()
        .with_metrics(["categorical_accuracy"])
        .enable_sparse_inference()
    )

    overfitted_accuracy = overfitted_model.predict(
        test_data, test_labels, predict_config
    )[0]["categorical_accuracy"]
    
    early_stop_accuracy = early_stop_model.predict(
        test_data, test_labels, predict_config
    )[0]["categorical_accuracy"]

    print(early_stop_accuracy, overfitted_accuracy)
    assert early_stop_accuracy >= overfitted_accuracy
