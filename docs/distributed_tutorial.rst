Instructions to use Distributed Bolt
====================================



Starting a Ray Cluster
----------------------
Below Cluster Initialization would only work for Custom Clusters. For standard clusters like AWS, GCP, Azure and Aliyun, see https://docs.ray.io/en/latest/cluster/cloud.html.




Installations
--------------------
- First, install all the required modules by running the following command(to be done on all the nodes):
- Have the Distributed bolt module built on those nodes 
- Run the command: ``pip install -r requirements.txt``
- Check if ray path is included in system path by printing (run 'echo $PATH' on the terminal)
- Ray's default path is ``/home/$USER/.local/bin``
- If not included, run the command: ``export PATH=$PATH:/home/$USER/.local/bin``


Automatic Cluster Initialization
----------------------------------
- Fill in the cluster configuration YAML file(cluster_configuration.YAML in cluster_configuration_files): 
- Edit all the required fields in the cluster_configuration.YAML
- head_ip: Add the IP for the head node 
- workers_ip: Add the IP for all the worker nodes
- ssh_private_key: Uncomment if ssh passwordless logging is not there on the nodes 
- min_workers, max_workers: By default make them min_workers == max_workers == len(worker_ips)
- For starting the ray cluster automatically, run the command: python3 start_cluster.py cluster_configuration.yaml
- For stopping the ray cluster automatically, run the command: python3 stop_cluster.py cluster_configuration.yaml
                
                
Manual Cluster Initialization
------------------------------
- On the head node, run the command ``ray start --head --port=6379``
- On worker nodes, run the command ``ray start --address=<head_node_ip>:6379	``
               


Validating Cluster
---------------------
- Type ``ray status`` on terminal
- See Node Status and Resources
- If it throws an error, it implies cluster has not started

Starting Training
-------------------
Make sure to divide the training data equally(almost) for all nodes. Otherwise, some batches might be dropped as the training will occur only on a minimum of the number of batches across nodes. 


Importing the library:

>>> from thirdai.distributed_bolt import db

The current APIs supported:

>>> head = db.FullyConnectedNetwork(
        num_workers=num_worker,
        config_filename=config_filename,
        num_cpus_per_node=num_cpus,
        communication_type="circular"/"linear",
    ) 
>>> head.train() 
>>> head.predict() #returns predict on the model trained

Look at examples folder for sample code.

IMPORTANT
------------------
1. Set up passwordless ssh between nodes(for easier usage)
2. If the num_cpus_per_node is not set, DistributedBolt will automatically get the number of CPUs available on the current and initialize the worker node with that count.
