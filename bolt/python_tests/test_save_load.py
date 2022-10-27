import pytest
from thirdai import bolt

from utils import gen_numpy_training_data

pytestmark = [pytest.mark.unit]


def get_train_config(epochs, batch_size):
    return bolt.TrainConfig(learning_rate=0.001, epochs=epochs).silence()


def get_predict_config():
    return bolt.PredictConfig().with_metrics(["categorical_accuracy"]).silence()


class ModelWithLayers:
    def __init__(self, n_classes):
        self.input_layer = bolt.nn.Input(dim=n_classes)

        self.hidden1 = bolt.nn.FullyConnected(
            dim=2000, sparsity=0.15, activation="relu"
        )(self.input_layer)

        self.hidden2 = bolt.nn.FullyConnected(
            dim=2000, sparsity=0.15, activation="relu"
        )(self.input_layer)

        self.concat = bolt.nn.Concatenate()([self.hidden1, self.hidden2])

        self.output = bolt.nn.FullyConnected(dim=n_classes, activation="softmax")(
            self.concat
        )

        self.model = bolt.nn.Model(inputs=[self.input_layer], output=self.output)

        self.model.compile(loss=bolt.nn.losses.CategoricalCrossEntropy())

    def train(self, data, labels, epochs):
        self.model.train(
            data, labels, train_config=get_train_config(epochs, batch_size=100)
        )

    def predict(self, data, labels):
        return self.model.predict(data, labels, predict_config=get_predict_config())[0]


def test_save_load_dag():
    n_classes = 100

    data, labels = gen_numpy_training_data(n_classes=n_classes, n_samples=10000)

    model = ModelWithLayers(n_classes=n_classes)

    # Train model and get accuracy.
    model.train(data, labels, epochs=1)
    test_metrics1 = model.predict(data, labels)
    assert test_metrics1["categorical_accuracy"] >= 0.9

    # Save and load as new model.
    save_loc = "./saved_dag_pymodel"
    model.model.save(filename=save_loc)
    new_model = bolt.nn.Model.load(filename=save_loc)

    # Verify accuracy matches.
    test_metrics2 = new_model.predict(
        data, labels, predict_config=get_predict_config()
    )[0]
    assert test_metrics2["categorical_accuracy"] >= 0.9
    assert (
        test_metrics1["categorical_accuracy"] == test_metrics2["categorical_accuracy"]
    )

    # Verify we can train the new model. Ideally we could check accuracy can
    # improve, but that is a bit flaky.
    new_model.train(
        data, labels, train_config=get_train_config(epochs=2, batch_size=100)
    )
    test_metrics3 = new_model.predict(
        data, labels, predict_config=get_predict_config()
    )[0]
    assert test_metrics3["categorical_accuracy"] >= 0.9


def test_save_fully_connected_layer_parameters():
    n_classes = 100

    data, labels = gen_numpy_training_data(n_classes=n_classes, n_samples=10000)

    model = ModelWithLayers(n_classes=n_classes)

    # Train model and get accuracy.
    model.train(data, labels, epochs=1)
    test_metrics1 = model.predict(data, labels)
    assert test_metrics1["categorical_accuracy"] >= 0.9

    # Save and load as new model.
    hidden1_save_loc = "./saved_dag_pymodel_hidden1"
    hidden2_save_loc = "./saved_dag_pymodel_hidden2"
    output_save_loc = "./saved_dag_pymodel_output"

    model.hidden1.save_parameters(hidden1_save_loc)
    model.hidden2.save_parameters(hidden2_save_loc)
    model.output.save_parameters(output_save_loc)

    new_model = ModelWithLayers(n_classes=n_classes)
    new_model.hidden1.load_parameters(hidden1_save_loc)
    new_model.hidden2.load_parameters(hidden2_save_loc)
    new_model.output.load_parameters(output_save_loc)

    # Verify accuracy matches.
    test_metrics2 = new_model.predict(data, labels)
    assert test_metrics2["categorical_accuracy"] >= 0.9
    assert (
        test_metrics1["categorical_accuracy"] == test_metrics2["categorical_accuracy"]
    )

    # Verify we can train the new model. Ideally we could check accuracy can
    # improve, but that is a bit flaky.
    new_model.train(data, labels, epochs=2)
    test_metrics3 = new_model.predict(data, labels)
    assert test_metrics3["categorical_accuracy"] >= 0.9
