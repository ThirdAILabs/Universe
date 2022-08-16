from ..test_mnist import load_mnist
from ..utils import get_simple_dag_model
from thirdai import bolt
import os
import pytest

# Add an integration test marker for all tests in this file
pytestmark = [pytest.mark.integration]


LEARNING_RATE = 0.0001


def setup_module():
    if not os.path.exists("mnist"):
        os.system(
            "curl https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.bz2 --output mnist.bz2"
        )
        os.system("bzip2 -d mnist.bz2")

    if not os.path.exists("mnist.t"):
        os.system(
            "curl https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.t.bz2 --output mnist.t.bz2"
        )
        os.system("bzip2 -d mnist.t.bz2")


def test_bolt_dag_on_mnist():
    model = get_simple_dag_model(
        input_dim=784, hidden_layer_dim=20000, hidden_layer_sparsity=0.01, output_dim=10
    )

    train_data, train_labels, test_data, test_labels = load_mnist()

    train_config = (
        bolt.graph.TrainConfig.make(learning_rate=0.0001, epochs=3)
        .silence()
        .with_rebuild_hash_tables(3000)
        .with_reconstruct_hash_functions(10000)
    )

    metrics = model.train(
        train_data=train_data, train_labels=train_labels, train_config=train_config
    )

    predict_config = (
        bolt.graph.PredictConfig.make().with_metrics(["categorical_accuracy"]).silence()
    )

    metrics = model.predict(
        test_data=test_data, test_labels=test_labels, predict_config=predict_config
    )

    assert metrics[0]["categorical_accuracy"] >= 0.9
