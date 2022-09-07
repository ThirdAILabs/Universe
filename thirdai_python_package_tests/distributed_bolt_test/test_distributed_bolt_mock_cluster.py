from ray.cluster_utils import Cluster
import pytest
import thirdai.distributed_bolt as db


pytestmark = [pytest.mark.xfail]

def test_distributed_bolt_on_mock_cluster():
    cluster = Cluster(
    initialize_head=True,
    head_node_args={
        "num_cpus": 1,
    })
    cluster.add_node(num_cpus=1)
    config_filename = "./default_config.txt"
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

    