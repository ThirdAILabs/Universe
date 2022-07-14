import os
import subprocess
import sys
import textwrap
import yaml
import getpass
import ray

def init_cluster_config():

    config = {'cluster_name': 'default', 'min_workers': 1, 'initial_workers': 1, 'max_workers': 2, 'autoscaling_mode': 'default', 'target_utilization_fraction': 0.8, 'idle_timeout_minutes': 5,
     'provider': {'type': 'local', 'head_ip': None, 'worker_ips': []}, 'auth': {'ssh_user': ''}, 'head_node': {}, 'worker_nodes': {}, 'file_mounts': {}, 'initialization_commands': [],
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
    

   
    config = init_cluster_config()
    print(nodes_ips)
    yaml_file = 'setup.yaml'
    with open(yaml_file, 'w') as file:
        user = getpass.getuser()
        
        config['auth']['ssh_user'] = user
        config['provider']['head_ip'] = nodes_ips[0]
        config['min_workers'] = 1
        config['max_workers'] = len(nodes_ips)

        if len(nodes_ips) > 1:
            config['provider']['worker_ips'] = nodes_ips[1:len(nodes_ips)]
        else:
            Warning(textwrap.dedent("""
                No worker ips provided.
            """))
        yaml.dump(config, file)
    

    system_path = os.environ['PATH']
    install_environment_locations = system_path.split(':')
    required_installation_path = '/home/' + user + '/.local/bin'

    if required_installation_path not in install_environment_locations:
        print(required_installation_path + ' not in System PATH')
        print('Updating PATH for ray')
        os.environ['PATH'] = required_installation_path + ':' + system_path
        os.system("export PATH=$PATH:/home/$USER/.local/bin")
        updated_system_path = os.environ['PATH']
        print('$PATH updated to', updated_system_path)
    else:
        print('Ray module already in system path.')

    print('Starting Ray Cluster')
    
    if not os.path.isdir('/tmp/ray'):
        print('Ray have not been initialised on this node before.')
        ray.init()
    
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