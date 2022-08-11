====================
Instructions to use Distributed Bolt
====================



Starting a Ray Cluster
--------------------
Below Cluster Initialization would only work for Custom Clusters, for standard clusters like AWS, GCP, Azure and Aliyun, see: https://docs.ray.io/en/latest/cluster/cloud.html.




Installations
--------------------
        - First install all the required module by running the following command(to be done on all the node):
        - Have the Distributed bolt module built on those node 
        - Run the command: pip install -r requirements.txt
        - Check if ray path is included in system path by printing (run 'echo $PATH' on terminal)
        - Ray's default path is /ome/$USER/.local/bin
        - If not included, run the command: export PATH=$PATH:/home/$USER/.local/bin
                
Automatic Cluster Initialization:
--------------------
        - Fill in the cluster configuration YAML file(cluster_configuration.yaml in cluster_configuration_files): 
        - Edit all the required field in the cluster_configuration.yaml
        - head_ip: Add the IP for head node 
        - workers_ip: Add the IP for all the worker nodes
        - ssh_private_key: Uncomment if ssh passwordless loging is not there on the nodes main_workers, max_workers: By default make them min_workers == max_workers == len(worker_ips)
        - For starting the ray cluster automatically, run the command: python3 start_cluster.py cluster_configuration.yaml
        - For stopping the ray cluster automatically, run the coammand: python3 stop_cluster.py cluster_configuration.yaml
                
                
Manual Cluster Initialization:
-------------------
        - On head node, run the command 'ray start --head --port=6379'
        - On worker nodes, run the command 'ray start --address=<head_node_ip>:6379'
               


Validating Cluster:
-------------------
        - Type 'ray status' on terminal
        - See Node Status and Reesources
        - If it throws an error, implies cluster has not started

Starting Training:
-------------------
        Make sure to divide the training data equally(almost) for all the node. Otherwise some batches might be dropped as the training will occur only on minimum of number of batches across nodes. 


        Importing the library:
        >>> from thirdai.distributed_bolt import DistributedBolt
        The current APIs supported:
        >>> head = DistributedBolt(num_of_workers=num_of_workers, config_filename=config_filename) 
        >>> head.train(circular=True, num_cpus_per_node=k(set number of cpus here manually)) 
        >>> print(head.predict()) #returns predict on the model trained

        Look at train_distributed_amzn670k.py for sample code.

IMPORTANT
------------------
        1. head_node_ip should be the first input of ray cluster
        2. Make sure you have the DistributedBolt_V1 branch is built on everynode(head node and worker node) you are running.
        3. Set up password less ssh between nodes(for easier usage)
        4. If the num_cpus_per_node is not set, DistributedBolt will automatically get the number of cpus available on the current and initialize the worker node with that count.