# Here we are mocking a cluster on a single machine
# without explicitly starting a ray cluster. We are
# testing both the communication circular and linear
# in the following tests.
# For reference: https://docs.ray.io/en/latest/ray-core/examples/testing-tips.html#tip-3-create-a-mini-cluster-with-ray-cluster-utils-cluster


import sys
import pytest
from thirdai import bolt, dataset

pytestmark = [pytest.mark.distributed]


try:
    import thirdai.distributed_bolt as db
    from ray.cluster_utils import Cluster
    import ray
except ImportError:
    pass


# This is some duplicate code with the mnist loading in test_mnist, but
# putting it somewhere both tests can read is more trouble than just having
# the loading code twice
def setup_module():
    import os

    path = "mnist_data"
    if not os.path.exists(path):
        os.makedirs(path)

    if not os.path.exists("mnist_data/xaa") or not os.path.exists("mnist_data/xab"):
        os.system(
            "curl https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.bz2 --output mnist.bz2"
        )
        os.system("bzip2 -d mnist.bz2")
        os.system("split -l 30000 mnist")

        os.system("rm mnist")
        os.system("mv xaa xab mnist_data/")

    if not os.path.exists("mnist_data/mnist.t"):
        os.system(
            "curl https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.t.bz2 --output mnist.t.bz2"
        )
        os.system("bzip2 -d mnist.t.bz2")
        os.system("mv mnist.t mnist_data/")


def get_mnist_model():
    input_layer = bolt.graph.Input(dim=784)

    hidden_layer = bolt.graph.FullyConnected(dim=256, sparsity=0.5, activation="Relu")(
        input_layer
    )

    output_layer = bolt.graph.FullyConnected(dim=10, activation="softmax")(hidden_layer)

    model = bolt.graph.Model(inputs=[input_layer], output=output_layer)

    model.compile(loss=bolt.CategoricalCrossEntropyLoss())

    return model


@pytest.fixture(scope="module")
def train_distributed_bolt_check(request):
    # Initilizing a mock cluster with two node
    mini_cluster = Cluster(
        initialize_head=True,
        head_node_args={
            "num_cpus": 1,
        },
    )
    mini_cluster.add_node(num_cpus=1)

    model = get_mnist_model()
    dataset_paths = ["mnist_data/xaa", "mnist_data/xab"]
    train_config = bolt.graph.TrainConfig.make(learning_rate=0.0001, epochs=1)
    cluster_config = db.RayTrainingClusterConfig(
        num_workers=2,
        requested_cpus_per_node=1,
        communication_type=request.param,
        cluster_address=mini_cluster.address,
    )

    distributed_model = db.DistributedDataParallel(
        cluster_config=cluster_config,
        model=model,
        train_config=train_config,
        train_file_names=dataset_paths,
        batch_size=256,
    )
    distributed_model.train()

    model_node_1 = distributed_model.get_model(worker_id=0)
    model_node_2 = distributed_model.get_model(worker_id=1)

    predict_config = (
        bolt.graph.PredictConfig.make().with_metrics(["categorical_accuracy"]).silence()
    )
    test_data, test_labels = dataset.load_bolt_svm_dataset(
        "mnist_data/mnist.t", batch_size=256
    )
    metrics_node_1 = model_node_1.predict(
        test_data=test_data, test_labels=test_labels, predict_config=predict_config
    )
    metrics_node_2 = model_node_2.predict(
        test_data=test_data, test_labels=test_labels, predict_config=predict_config
    )

    # This is a weak check that the parameters are the same on the two nodes
    assert (
        abs(
            metrics_node_1[0]["categorical_accuracy"]
            - metrics_node_2[0]["categorical_accuracy"]
        )
        < 0.001
    )

    ray.shutdown()
    mini_cluster.shutdown()

    yield metrics_node_1


@pytest.mark.skipif("ray" not in sys.modules, reason="requires the ray library")
@pytest.mark.parametrize(
    "train_distributed_bolt_check", ["linear", "circular"], indirect=True
)
def test_distributed_bolt_on_mock_cluster(train_distributed_bolt_check):
    import multiprocessing

    if multiprocessing.cpu_count() < 2:
        assert False, "not enough cpus for distributed training"

    assert train_distributed_bolt_check[0]["categorical_accuracy"] > 0.9
