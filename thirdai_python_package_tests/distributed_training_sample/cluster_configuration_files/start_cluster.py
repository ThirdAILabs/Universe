import os
import subprocess
import sys
import textwrap
import yaml
import getpass
import ray
import warnings


def start_cluster(config_yaml_file) -> None:
    """Start a ray cluster with the node ips provided.

    Args:
        node_ips: List of node ips to start cluster on.
        The list must be in this order:
        [<head_node_ip> <worker1_node_ip> <worker2_node_ip> <worker3_node_ip> ....]

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
            warnings.warn(
                textwrap.dedent(
                    """
                min_workers >= 0
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
    required_installation_path = "/home/" + user + "/.local/bin"

    if required_installation_path not in install_environment_locations:
        print(required_installation_path + " not in System PATH")
        current_path = os.environ["PATH"]
        print("Current Path: ", current_path)
        print("Run the command: export PATH=$PATH:/home/$USER/.local/bin")
    else:
        print("Ray module already in system path.")

    print("Starting Ray Cluster")

    if not os.path.isdir("/tmp/ray"):
        print("It look like ray is never used here before!")
        ray.init()

    os.system("ray stop")
    os.system("ray up cluster_configuration.yaml")


if __name__ == "__main__":
    config_yaml_file = sys.argv[1]

    start_cluster(config_yaml_file)
