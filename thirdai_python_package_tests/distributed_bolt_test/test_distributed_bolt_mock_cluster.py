# Here we are mocking a cluster on a single machine
# without explicitly starting a ray cluster. We are
# testing both the communication circular and linear
# in the following tests.
# For reference: https://docs.ray.io/en/latest/ray-core/examples/testing-tips.html#tip-3-create-a-mini-cluster-with-ray-cluster-utils-cluster


import sys
import pytest
import os
import multiprocessing


try:
    import thirdai.distributed_bolt as db
    from ray.cluster_utils import Cluster
    import ray
except ImportError:
    pass


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


@pytest.fixture(scope="module")
def train_distributed_bolt_check(request):
    # Initilizing a mock cluster with two node
    cluster = Cluster(
        initialize_head=True,
        head_node_args={
            "num_cpus": 1,
        },
    )
    cluster.add_node(num_cpus=1)

    # Configuration file the training
    config_filename = os.path.join(
        os.path.dirname(__file__),  # Directory where this .py file is
        "default_config.txt",
    )

    if ray.is_initialized():
        ray.shutdown()

    head = db.FullyConnectedNetwork(
        num_workers=2,
        config_filename=config_filename,
        num_cpus_per_node=1,
        communication_type=request.param,
        cluster_address=cluster.address,
    )
    head.train()
    metrics = head.predict()

    # shutting down the ray and cluster
    ray.shutdown()
    cluster.shutdown()

    yield metrics


@pytest.mark.skipif("ray" not in sys.modules, reason="requires the ray library")
@pytest.mark.xfail
@pytest.mark.parametrize(
    "train_distributed_bolt_check", ["linear", "circular"], indirect=True
)
def test_distributed_bolt_on_mock_cluster(train_distributed_bolt_check):
    if multiprocessing.cpu_count() < 2:
        assert False, "not enough cpus for distributed training"

    assert train_distributed_bolt_check[0]["categorical_accuracy"] > 0.9
