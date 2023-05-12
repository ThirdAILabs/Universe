import pytest
from download_dataset_fixtures import download_mnist_dataset
from thirdai import bolt, dataset

# Add an integration test marker for all tests in this file
pytestmark = [pytest.mark.unit]


LEARNING_RATE = 0.0001


@pytest.fixture
def load_mnist(download_mnist_dataset):
    train_file, test_file = download_mnist_dataset
    train_x, train_y = dataset.load_bolt_svm_dataset(train_file, 250)
    test_x, test_y = dataset.load_bolt_svm_dataset(test_file, 250)
    return train_x, train_y, test_x, test_y


def test_bolt_dag_on_mnist(load_mnist):
    input_layer = bolt.nn.Input(dim=784)

    hidden_layer = bolt.nn.FullyConnected(
        dim=20000,
        sparsity=0.01,
        activation="relu",
        sampling_config=bolt.nn.DWTASamplingConfig(
            num_tables=64,
            hashes_per_table=3,
            range_pow=9,
            binsize=8,
            reservoir_size=32,
            permutations=8,
        ),
    )(input_layer)

    output_layer = bolt.nn.FullyConnected(dim=10, activation="softmax")(hidden_layer)

    model = bolt.nn.Model(inputs=[input_layer], output=output_layer)

    model.compile(loss=bolt.nn.losses.CategoricalCrossEntropy())

    train_data, train_labels, test_data, test_labels = load_mnist

    train_config = (
        bolt.TrainConfig(learning_rate=0.0001, epochs=3)
        .silence()
        .with_rebuild_hash_tables(3000)
        .with_reconstruct_hash_functions(10000)
    )

    metrics = model.train(
        train_data=train_data, train_labels=train_labels, train_config=train_config
    )

    eval_config = bolt.EvalConfig().with_metrics(["categorical_accuracy"]).silence()

    metrics = model.evaluate(
        test_data=test_data, test_labels=test_labels, eval_config=eval_config
    )

    assert metrics[0]["categorical_accuracy"] >= 0.9
