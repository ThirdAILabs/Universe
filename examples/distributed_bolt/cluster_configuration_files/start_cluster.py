import getpass
import os
import subprocess
import sys
import textwrap
import warnings
from ast import Import

import ray
import yaml


def start_cluster(config_yaml_file) -> None:
    """Start a ray cluster with the node ips provided.

    Args:
        config_yaml_file: Configuration file for starting a cluster.

    """

    yaml_file = config_yaml_file
    user = getpass.getuser()
    with open(yaml_file, "r") as file:
        config = yaml.safe_load(file)
        if "ssh_private_key" not in config["auth"]:
            warnings.warn(
                textwrap.dedent(
                    """
                ssh_private_key field, not given. Make sure
                you have the id_rsa copied on all the nodes.
                Else uncomment the ssh_private_key
            """
                )
            )

        number_of_worker_nodes = len(config["provider"]["worker_ips"])
        if config["min_workers"] < 0:
            raise ValueError(
                textwrap.dedent(
                    """
                min_workers should be >= 0.
            """
                )
            )

        if config["max_workers"] < number_of_worker_nodes:
            warnings.warn(
                textwrap.dedent(
                    """
                max_worker field in config is less than
                number of worker node provided. Autoscalar 
                would only start number of worker reported in
                max_workers field. Other worker need to be initialized
                manually.
            """
                )
            )

    system_path = os.environ["PATH"]
    install_environment_locations = system_path.split(":")
    required_installation_path = os.popen("which ray").read().replace("/ray\n", "")

    if required_installation_path == "":
        raise ImportError("No module name ray, Try doing pip install 'ray[default]'")
    elif required_installation_path not in install_environment_locations:
        print(required_installation_path + " not in System PATH")
        current_path = os.environ["PATH"]
        print("Current Path: ", current_path)
        raise ValueError("Run the command: export PATH=$PATH:/home/$USER/.local/bin")
    else:
        print("Ray module already in system path.")

    if not os.path.isdir("/tmp/ray"):
        print("It look like ray is never used here before!")
        ray.init()

    os.system("ray stop")
    os.system("ray up " + config_yaml_file)


if __name__ == "__main__":
    config_yaml_file = sys.argv[1]

    start_cluster(config_yaml_file)
