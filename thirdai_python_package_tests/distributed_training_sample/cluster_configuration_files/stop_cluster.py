import os
import subprocess
import sys
import textwrap
import yaml
import getpass
import ray
import warnings


def stop_cluster(config_yaml_file) -> None:
    """Start a ray cluster with the node ips provided.

    Args:
        node_ips: List of node ips to start cluster on.
        The list must be in this order:
        [<head_node_ip> <worker1_node_ip> <worker2_node_ip> <worker3_node_ip> ....]

    """

    os.system("ray down " + config_yaml_file)


if __name__ == "__main__":
    config_yaml_file = sys.argv[1]

    stop_cluster(config_yaml_file)
