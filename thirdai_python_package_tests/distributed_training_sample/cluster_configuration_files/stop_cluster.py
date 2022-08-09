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
        config_yaml_file: Configuration File for stopping the clster.
        (File should be same as the file which started the cluster.)

    """

    os.system("ray down " + config_yaml_file)


if __name__ == "__main__":
    config_yaml_file = sys.argv[1]

    stop_cluster(config_yaml_file)
