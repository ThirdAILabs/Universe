import os

import numpy as np
import pytest
from thirdai import dataset
from thirdai.demos import download_mnist_dataset
from thirdai import bolt_v2 as bolt


@pytest.fixture(scope="module")
def ray_two_node_cluster_config():
    # Do these imports here so pytest collection doesn't fail if ray isn't installed
    import ray
    import thirdai.distributed_bolt as db
    from ray.cluster_utils import Cluster

    num_cpu_per_node = db.get_num_cpus() // 2

    # case if multiprocessing import fails
    if num_cpu_per_node == 0:
        num_cpu_per_node = 1

    mini_cluster = Cluster(
        initialize_head=True,
        head_node_args={
            "num_cpus": num_cpu_per_node,
        },
    )
    mini_cluster.add_node(num_cpus=num_cpu_per_node)

    # directly yielding mini_cluster returns a generator for cluster_config,
    # rather than cluster_config itself and those generators were just using
    # the default communication_type(= "linear"), even after parametrizing it
    # . doing it this way make sure we are getting the cluster_config for the
    # communication type provided
    def _make_cluster_config(communication_type="linear"):
        # We set the working_dir for the cluster equal to this directory
        # so that pickle works. Otherwise, unpickling functions
        # defined in the test files would not work, since pickle needs to be
        # able to import the file the object/function was originally defined in.

        working_dir = os.path.dirname(os.path.realpath(__file__))
        cluster_config = db.RayTrainingClusterConfig(
            num_workers=2,
            requested_cpus_per_node=num_cpu_per_node,
            communication_type=communication_type,
            cluster_address=mini_cluster.address,
            runtime_env={"working_dir": working_dir},
            ignore_reinit_error=True,
        )
        return cluster_config

    yield _make_cluster_config

    ray.shutdown()
    mini_cluster.shutdown()


def split_into_2(
    file_to_split, destination_file_1, destination_file_2, with_header=False
):
    with open(file_to_split, "r") as input_file:
        with open(destination_file_1, "w+") as f_1:
            with open(destination_file_2, "w+") as f_2:
                for i, line in enumerate(input_file):
                    if with_header and i == 0:
                        f_1.write(line)
                        f_2.write(line)
                        continue

                    if i % 2 == 0:
                        f_1.write(line)
                    else:
                        f_2.write(line)


def compare_parameters_of_two_models(model_node_1, model_node_2, atol=1e-8):
    nodes_1 = model_node_1.nodes()
    nodes_2 = model_node_2.nodes()
    for layer_1, layer_2 in zip(nodes_1, nodes_2):
        if hasattr(layer_1, "weights"):
            assert np.allclose(layer_1.weights.get(), layer_2.weights.get(), atol=atol)
        if hasattr(layer_1, "biases"):
            assert np.allclose(layer_1.biases.get(), layer_2.biases.get(), atol=atol)


def check_models_are_same_on_first_two_nodes(distributed_model):
    model_node_1 = distributed_model.get_model(worker_id=0)
    model_node_2 = distributed_model.get_model(worker_id=1)

    compare_parameters_of_two_models(model_node_1, model_node_2)


def remove_files(file_names):
    for file in file_names:
        if os.path.exists(file):
            os.remove(file)


def metrics_aggregation_from_workers(train_metrics):
    overall_metrics = {}
    for metrics_per_node in train_metrics:
        for key, value in metrics_per_node.items():
            if key not in overall_metrics:
                overall_metrics[key] = 0
            # Here we are averaging the metrics, hence divding the
            # metric "categorical_accuracy" by 2(we use only two
            # workers for testing purpose).
            overall_metrics[key] += value[-1] / 2

    return overall_metrics


@pytest.fixture(scope="session")
def mnist_distributed_split(download_mnist_dataset):
    train_file, test_file = download_mnist_dataset
    path = "mnist_data"
    if not os.path.exists(path):
        os.makedirs(path)

    split_into_2(
        file_to_split=train_file,
        destination_file_1="mnist_data/part1",
        destination_file_2="mnist_data/part2",
    )

    return ("mnist_data/part1", "mnist_data/part2"), test_file


def gen_numpy_training_data(
    n_classes=10,
    n_samples=1000,
    noise_std=0.1,
    convert_to_bolt_dataset=True,
    batch_size_for_conversion=64,
):
    possible_one_hot_encodings = np.eye(n_classes)
    labels = np.random.choice(n_classes, size=n_samples).astype("uint32")
    examples = possible_one_hot_encodings[labels]
    noise = np.random.normal(0, noise_std, examples.shape)
    examples = (examples + noise).astype("float32")
    if convert_to_bolt_dataset:
        examples = dataset.from_numpy(examples, batch_size=batch_size_for_conversion)
        labels = dataset.from_numpy(labels, batch_size=batch_size_for_conversion)
    return examples, labels


def check_model_parameters_equal(model_0, model_1):
    for op_0, op_1 in zip(model_0.ops(), model_1.ops()):
        assert np.allclose(op_0.weights, op_1.weights)
        assert np.allclose(op_0.biases, op_1.biases)


def check_model_parameters_match(distributed_model):
    model_0 = distributed_model.get_model(0)
    model_1 = distributed_model.get_model(1)

    check_model_parameters_equal(model_0, model_1)


def get_bolt_model():
    n_classes = 10
    input_layer = bolt.nn.Input(dim=n_classes)

    hidden_layer = bolt.nn.FullyConnected(
        dim=20000,
        input_dim=n_classes,
        sparsity=0.01,
        activation="relu",
        rebuild_hash_tables=12,
        reconstruct_hash_functions=40,
    )(input_layer)
    output = bolt.nn.FullyConnected(
        dim=n_classes, input_dim=20000, activation="softmax"
    )(hidden_layer)

    labels = bolt.nn.Input(dim=n_classes)
    loss = bolt.nn.losses.CategoricalCrossEntropy(output, labels)

    model = bolt.nn.Model(inputs=[input_layer], outputs=[output], losses=[loss])

    return model
