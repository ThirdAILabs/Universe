import os
import subprocess
import sys
import textwrap
import yaml

def init_cluster_config():

    config = {'cluster_name': 'default', 'min_workers': 1, 'initial_workers': 1, 'max_workers': 2, 'autoscaling_mode': 'default', 'target_utilization_fraction': 0.8, 'idle_timeout_minutes': 5,
     'provider': {'type': 'local', 'head_ip': None, 'worker_ips': []}, 'auth': {'ssh_user': 'pratik'}, 'head_node': {}, 'worker_nodes': {}, 'file_mounts': {}, 'initialization_commands': [],
      'setup_commands': ["pip install 'ray[default]'"], 'head_setup_commands': [], 'worker_setup_commands': [], 'head_start_ray_commands': ['ray stop', 'ray start --head --port=6379'],
       'worker_start_ray_commands': ['ray stop', 'ray start --address=$RAY_HEAD_IP:6379']}
    return config


def start_cluster(
    nodes_ips
) -> None:
    """ Start a ray cluster with the node ips provided.

        Args:
            node_ips: List of node ips to start cluster on.
            The list must be in this order:
            [<head_node_ip> <worker1_node_ip> <worker2_node_ip> <worker3_node_ip> ....]
                
    """
    
    os.system("pip3 install --no-cache-dir ray[default]")
    os.system("pip3 install --no-cache-dir pyyaml")

   
    config = init_cluster_config()
    print(nodes_ips)
    yaml_file = 'setup.yaml'
    with open(yaml_file, 'w') as file:
        config['provider']['head_ip'] = nodes_ips[0]
        if len(nodes_ips) > 1:
            config['provider']['worker_ips'] = nodes_ips[1:len(nodes_ips)]
        else:
            Warning(textwrap.dedent("""
                No worker ips provided.
            """))
        yaml.dump(config, file)
    
    os.system("export PATH=$PATH:/home/$USER/.local/bin")
    os.system("ray stop")
    os.system('ray up setup.yaml')
    os.system('rm setup.yaml')

    connect_address = nodes_ips[0] + ":6379"
    subprocess.run(["sh", "make_cluster.sh", " ".join(nodes_ips[1:len(nodes_ips)]), connect_address])

if __name__ == "__main__":
    total_nodes = len(sys.argv) - 1

    if total_nodes == 0:
        raise Exception(textwrap.dedent("""
            No ips has been passed as an argument.
            Pass nodes ip as argument to this python file.
            Run "python3 start_cluster.py <head_node_ip> <worker1_node_ip> <worker2_node_ip> <worker3_node_ip> ...." 
        """))
    
    node_ip_list = []
    for i in range(1, total_nodes+1):
        node_ip_list.append(sys.argv[i])
    
    start_cluster(node_ip_list)