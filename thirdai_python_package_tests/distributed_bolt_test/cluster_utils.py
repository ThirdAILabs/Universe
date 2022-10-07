import pytest

try:
    import ray
    import thirdai.distributed_bolt as db
    from ray.cluster_utils import Cluster
except ImportError:
    pass
import math
import os

import numpy as np


@pytest.fixture(scope="module")
def ray_two_node_cluster_config(communication_type="linear"):
    mini_cluster = Cluster(
        initialize_head=True,
        head_node_args={
            "num_cpus": 1,
        },
    )
    mini_cluster.add_node(num_cpus=1)

    # We set the working_dir for the cluster equal to this directory
    # so that pickle works. Otherwise, unpickling the function
    # defined in test_mock_cluster_arbitrary_streaming_data_loader.py would not
    # work, since pickle needs to be able to import the file the object/function
    # was originally defined in.
    working_dir = os.path.dirname(os.path.realpath(__file__))
    cluster_config = db.RayTrainingClusterConfig(
        num_workers=2,
        requested_cpus_per_node=1,
        communication_type=communication_type,
        cluster_address=mini_cluster.address,
        runtime_env={"working_dir": working_dir},
        ignore_reinit_error=True,
    )
    yield cluster_config

    ray.shutdown()
    mini_cluster.shutdown()


def split_into_2(file_to_split, destination_dir):
    num_lines = sum(1 for _ in open(file_to_split))
    os.system(f"split -l {math.ceil(num_lines/2)} {file_to_split}")

    os.system(f"rm {file_to_split}")
    os.system(f"mv xaa xab {destination_dir}/")


def check_models_are_same_on_first_two_nodes(distributed_model):
    model_node_1 = distributed_model.get_model(worker_id=0)
    model_node_2 = distributed_model.get_model(worker_id=1)

    nodes_1 = model_node_1.nodes()
    nodes_2 = model_node_2.nodes()
    for layer_1, layer_2 in zip(nodes_1, nodes_2):
        if hasattr(layer_1, "weights"):
            assert np.allclose(layer_1.weights.get(), layer_2.weights.get())
