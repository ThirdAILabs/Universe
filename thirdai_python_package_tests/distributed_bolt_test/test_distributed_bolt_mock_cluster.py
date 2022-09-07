from ray.cluster_utils import Cluster
import pytest
import thirdai.distributed_bolt as db


pytestmark = [pytest.mark.unit]

def download_data():
    import os

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

    os.system('split -l 30000 mnist')


    path = 'mnist_data'
    if not os.path.exists(path):
        os.makedirs(path)

    os.system('rm mnist')
    os.system('cp xaa xab mnist.t mnist_data/')
    os.system('rm xaa xab mnist.t')


def test_distributed_bolt_on_mock_cluster():
    cluster = Cluster(
    initialize_head=True,
    head_node_args={
        "num_cpus": 1,
    })
    cluster.add_node(num_cpus=1)
    config_filename = "../thirdai_python_package_tests/distributed_bolt_test/default_config.txt"
    head = db.FullyConnectedNetwork(
        num_workers=2,
        config_filename=config_filename,
        num_cpus_per_node=1,
        communication_type="linear",
        cluster_address = cluster.address,
    )
    head.train()
    metrics = head.predict()
    print(metrics[0]["categorical_accuracy"])

    