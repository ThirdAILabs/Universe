from thirdai import bolt, dataset
import numpy as np
import ray
import os
import time as time 
import toml
from typing import Tuple, Any, Optional, Dict, List
import subprocess

# Make sure to add config file to working directory in environment variable


def create_fully_connected_layer_configs(
    configs: List[Dict[str, Any]]
) -> List[bolt.FullyConnected]:
    layers = []
    for config in configs:

        if config.get("use_default_sampling", False):
            layer = bolt.FullyConnected(
                dim=config.get("dim"),
                sparsity=config.get("sparsity", 1.0),
                activation_function=bolt.getActivationFunction(
                    config.get("activation")
                ),
            )
        else:
            layer = bolt.FullyConnected(
                dim=config.get("dim"),
                sparsity=config.get("sparsity", 1.0),
                activation_function=bolt.getActivationFunction(
                    config.get("activation")
                ),
                sampling_config=bolt.SamplingConfig(
                    hashes_per_table=config.get("hashes_per_table", 0),
                    num_tables=config.get("num_tables", 0),
                    range_pow=config.get("range_pow", 0),
                    reservoir_size=config.get("reservoir_size", 128),
                    hash_function=config.get("hash_function", "DWTA"),
                ),
            )

        layers.append(layer)
    return layers

def find_full_filepath(filename: str) -> str:
    data_path_file = ("./dataset_paths.toml")
    prefix_table = toml.load(data_path_file)
    for prefix in prefix_table["prefixes"]:
        if os.path.exists(prefix + filename):
            return prefix + filename
    print(
        "Could not find file '"
        + filename
        + "' on any filepaths. Add correct path to 'Universe/dataset_paths.toml'"
    )
    sys.exit(1)


def load_dataset(
    config: Dict[str, Any]
    , total_nodes) -> Optional[
        Tuple[
            dataset.BoltDataset,  # train_x
            dataset.BoltDataset,  # train_y
            dataset.BoltDataset,  # test_x
            dataset.BoltDataset,  # test_y
        ]
    ]:
    train_filename = find_full_filepath(config["dataset"]["train_data"])
    test_filename = find_full_filepath(config["dataset"]["test_data"])
    batch_size = int(config["params"]["batch_size"]/total_nodes)
    if config["dataset"]["format"].lower() == "svm":
        train_x, train_y = dataset.load_bolt_svm_dataset(train_filename, batch_size)
        test_x, test_y = dataset.load_bolt_svm_dataset(test_filename, batch_size)
        return train_x, train_y, test_x, test_y
    elif config["dataset"]["format"].lower() == "csv":
        delimiter = config["dataset"].get("delimeter", ",")
        train_x, train_y = dataset.load_bolt_csv_dataset(
            train_filename, batch_size, delimiter
        )
        test_x, test_y = dataset.load_bolt_csv_dataset(
            test_filename, batch_size, delimiter
        )
        return train_x, train_y, test_x, test_y
    else:
        print("Invalid dataset format specified")
        return None


@ray.remote(num_cpus=40, max_restarts=1)
class Worker:
    def __init__(self, layers, config, random_seed, total_nodes, id):
        self.layers = layers
        self.bolt_layers = create_fully_connected_layer_configs(config["layers"])
        self.input_dim = config["dataset"]["input_dim"]
        self.network = bolt.DistributedNetwork(layers=self.bolt_layers, input_dim=self.input_dim)
        self.rehash = config["params"]["rehash"]
        self.rebuild = config["params"]["rebuild"]
        use_sparse_inference = "sparse_inference_epoch" in config["params"].keys()
        if use_sparse_inference:
            sparse_inference_epoch = config["params"]["sparse_inference_epoch"]
        else:
            sparse_inference_epoch = None

        data = load_dataset(config, total_nodes)
        if data is None:
            raise ValueError("Unable to load a dataset. Please check the config")
        self.train_data, self.train_label, self.test_data, self.test_label = data
        self.num_of_batches = self.network.initTrainSingleNode(
                    self.train_data, 
                    self.train_label,
                    rehash=self.rehash,
                    rebuild=self.rebuild,
                    verbose=False,
                    random_seed=random_seed)
        if config["params"]["loss_fn"].lower() == "categoricalcrossentropyloss":
            self.loss = bolt.CategoricalCrossEntropyLoss()
        elif config["params"]["loss_fn"].lower() == "meansquarederror":
            self.loss = bolt.MeanSquaredError()
        else:
            print("'{}' is not a valid loss function".format(config["params"]["loss_fn"]))
        
        self.total_nodes = total_nodes
        self.id = id


    def addSupervisor(self, supervisor):
        self.supervisor = supervisor
        

    def addFriend(self, friend):
        self.friend = friend

    def calculateGradientsCircular(self, batch_no):
        self.network.calculateGradientSingleNode(batch_no, self.loss)
        w_gradients = []
        b_gradients = []
        self.w_partitions = []
        self.b_partitions = []
        for layer in range(len(self.layers)-1):
            x = self.network.get_weights_gradients(layer)
            y = self.network.get_biases_gradients(layer)
            w_gradients.append(x)
            b_gradients.append(y)
            self.w_partitions.append(int(len(x)/self.total_nodes))
            self.b_partitions.append(int(len(y)/self.total_nodes))
        self.w_gradients, self.b_gradients = w_gradients, b_gradients
        return True

    def calculateGradientsLinear(self, batch_no):
        self.network.calculateGradientSingleNode(batch_no, self.loss)
        w_gradients = []
        b_gradients = []
        for layer in range(len(self.layers)-1):
            x = self.network.get_weights_gradients(layer)
            y = self.network.get_biases_gradients(layer)
            w_gradients.append(x)
            b_gradients.append(y)
        ray_gradient_object = ray.put((w_gradients, b_gradients))
        return ray_gradient_object

    def returnParams(self):
        weights = []
        biases = []
        for layer in range(len(self.layers)-1):
            x = self.network.get_weights(layer)
            y = self.network.get_biases(layer)
            weights.append(x)
            biases.append(y)
        return weights, biases

    def receiveParams(self):
        weights, biases = ray.get(self.supervisor.weights_biases.remote())
        for layer in range(len(weights)):
            self.network.set_weights(layer, weights[layer])
            self.network.set_biases(layer, biases[layer])
        return True

    def receiveGradientsCircularCommunication(self):
        for layer in range(len(self.w_gradients)):
            self.network.set_weights_gradients(layer, self.w_gradients[layer])
            self.network.set_biases_gradients(layer, self.b_gradients[layer])
        return True

    def receiveGradientsLinearCommunication(self):
        w_gradients_updated, b_gradients_updated = ray.get(self.supervisor.gradients_avg.remote())
        for layer in range(len(w_gradients_updated)):
            self.network.set_weights_gradients(layer, w_gradients_updated[layer])
            self.network.set_biases_gradients(layer, b_gradients_updated[layer])
        return True

    def processRing(self, update_id, reduce = True, avg_gradients = False):
        local_update_id = (update_id + self.id - 1)%self.total_nodes


        get_ray_object = self.friend.receiveArrayPartitions.remote(update_id)
        self.friend_weight_gradient_list, self.friend_bias_gradient_list = ray.get(get_ray_object)

        for i in range(len(self.friend_weight_gradient_list)):
            l_weight_id = self.w_partitions[i] * local_update_id
            r_weight_id = self.w_partitions[i] * (local_update_id + 1)
            if len(self.w_gradients[i]) - r_weight_id < self.w_partitions[i]:
                r_weight_id = len(self.w_gradients[i])

            l_bias_id = self.b_partitions[i] * local_update_id
            r_bias_id = self.b_partitions[i] * (local_update_id + 1)
            if len(self.b_gradients[i]) - r_bias_id < self.w_partitions[i]:
                r_bias_id = len(self.b_gradients[i])

            assert self.w_partitions[i] > 0, f'weight partions has value {self.w_partitions[i]}'
            assert self.b_partitions[i] > 0, f'bias partions has value {self.b_partitions[i]}'
            assert r_weight_id-l_weight_id >= self.w_partitions[i], f'weight update index range are less than {self.w_partitions[i]}'
            assert r_bias_id-l_bias_id >= self.b_partitions[i], f'bias update index range are less than {self.b_partitions[i]}'


            # arrays should be numpy arrays for the following operation, otherwise it will just get appened to the list
            if reduce:
                self.w_gradients[i][l_weight_id:r_weight_id] += self.friend_weight_gradient_list[i]
                self.b_gradients[i][l_bias_id:r_bias_id] += self.friend_bias_gradient_list[i]
                if avg_gradients:
                    self.w_gradients[i][l_weight_id:r_weight_id] = self.w_gradients[i][l_weight_id:r_weight_id]/self.total_nodes
                    self.b_gradients[i][l_bias_id:r_bias_id] = self.b_gradients[i][l_bias_id:r_bias_id]/self.total_nodes
            else:
                self.w_gradients[i][l_weight_id:r_weight_id] = self.friend_weight_gradient_list[i]
                self.b_gradients[i][l_bias_id:r_bias_id] = self.friend_bias_gradient_list[i]

        return True

    

    def receiveArrayPartitions(self, update_id):
        local_update_id = (update_id + self.id)%self.total_nodes
        
        w_gradient_subarray = []
        b_gradient_subarray = []
        for i in range(len(self.w_partitions)):
            l_weight_id = self.w_partitions[i] * local_update_id
            r_weight_id = self.w_partitions[i] * (local_update_id + 1)
            if len(self.w_gradients[i]) - r_weight_id < self.w_partitions[i]:
                r_weight_id = len(self.w_gradients[i])

            l_bias_id = self.b_partitions[i] * local_update_id
            r_bias_id = self.b_partitions[i] * (local_update_id + 1)
            if len(self.b_gradients[i]) - r_bias_id < self.b_partitions[i]:
                r_bias_id = len(self.b_gradients[i])

            assert self.w_partitions[i] > 0, f'weight partions has value {self.w_partitions[i]}'
            assert self.b_partitions[i] > 0, f'bias partions has value {self.b_partitions[i]}'
            assert r_weight_id-l_weight_id >= self.w_partitions[i], f'weight update index range are less than {self.w_partitions[i]}'
            assert r_bias_id-l_bias_id >= self.b_partitions[i], f'bias update index range are less than {self.b_partitions[i]}'

            w_gradient_subarray.append(self.w_gradients[i][l_weight_id:r_weight_id])
            b_gradient_subarray.append(self.b_gradients[i][l_bias_id:r_bias_id])

        
        return w_gradient_subarray, b_gradient_subarray

    

    def updateParameters(self, learning_rate):
        self.network.updateParametersSingleNode(learning_rate)
        return True
        
    
    def num_of_batches(self):
        return self.num_of_batches

    def predict(self):
        acc, _ = self.network.predictSingleNode(
            self.test_data, self.test_label, self.num_of_batches, False, ["categorical_accuracy"], verbose=False
        )
        return acc["categorical_accuracy"]
    

@ray.remote(num_cpus=2, max_restarts=2)
class Supervisor:
    def __init__(self, layers, workers, config):
        self.layers = layers
        self.workers = workers
        self.num_of_batches = ray.get(self.workers[0].num_of_batches.remote())
        self.weights_biases = ray.get(self.workers[0].returnParams.remote())
        
    

    def batch_to_train(self, id, batch_no):
        return int(self.num_of_batches/len(self.workers))*id + batch_no

    def subworkCircularCommunication(self, batch_no):
        updates = ray.get([self.workers[id].calculateGradientsCircular.remote(self.batch_to_train(id, batch_no)) for id in range(len(self.workers))])

        # First Run
        update_id = 0
        for i in range(self.no_of_workers-1):
            if i == self.no_of_workers - 2:
                blocking_run = ray.get([w.processRing.remote(update_id, avg_gradients=True) for w in self.workers])
            else:
                blocking_run = ray.get([w.processRing.remote(update_id) for w in self.workers])
            update_id -= 1

        # Second Run 
        update_id = 1
        for i in range(self.no_of_workers-1):
            blocking_run = ray.get([w.processRing.remote(update_id, reduce=False) for w in self.workers])
            update_id -= 1
        return True

    def subworkLinearCommunication(self, batch_no):
        gradients_list = ray.get([self.workers[id].calculateGradientsLinear.remote(self.batch_to_train(id, batch_no)) for id in range(len(self.workers))])
        gradients_list = ray.get(gradients_list)

        w_gradients_avg = np.array([np.zeros((self.layers[layer_no+1], self.layers[layer_no])) for layer_no in range(len(self.layers)-1)])
        b_gradients_avg = np.array([np.zeros((self.layers[layer_no+1])) for layer_no in range(len(self.layers)-1)])


        for w_gradients,b_gradients in gradients_list:
            w_gradients_avg += w_gradients
            b_gradients_avg += b_gradients
        
        self.w_gradients_avg = w_gradients_avg/len(self.workers)
        self.b_gradients_avg = b_gradients_avg/len(self.workers)
        return True

    
    def gradients_avg(self):
        return self.w_gradients_avg, self.b_gradients_avg

    
    def subworkUpdateParameters(self, learning_rate):
        check_update_parameter = ray.get([w.updateParameters.remote(learning_rate) for w in self.workers])
        return True

    
    def check_weights(self):
        weights_0, biases_0 = ray.get(self.workers[0].returnParams.remote())
        weights_1, biases_1 = ray.get(self.workers[1].returnParams.remote())
        print('weights 0: ', weights_0)
        print('weights 1: ', weights_1)
        print('biases 0: ', biases_0)
        print('biases 1: ', biases_1)


    def weights_biases(self):
        return self.weights_biases
        
           
class DistributedBolt:
    def __init__(self, worker_nodes, config_filename):
        os.system("pip3 install --no-cache-dir ray[default]")
        os.system("export PATH=$PATH:/home/$USER/.local/bin")
        subprocess.run(["sh", "make_cluster.sh", " ".join(worker_nodes)])
        self.no_of_workers = len(worker_nodes)+1
        runtime_env = {"working_dir": "/share/pratik/RayRuntimeEnvironment", "pip": ["toml", "typing", "typing_extensions", 'psutil'], "env_vars": {"OMP_NUM_THREADS": "100"}}
        ray.init(address='auto', runtime_env=runtime_env)
        config = toml.load(config_filename)
        self.epochs = config["params"]["epochs"]
        self.learning_rate = config["params"]["learning_rate"]
        self.layers = [config["dataset"]["input_dim"]]
        for i in range(len(config['layers'])):
            self.layers.append(config['layers'][i]['dim'])
        random_seed = int(time.time())%int(0xffffffff)
        self.workers = [Worker.options(max_concurrency=2).remote(self.layers,config, random_seed, self.no_of_workers, id) for id in range(self.no_of_workers)]
        self.supervisor = Supervisor.remote(self.layers,self.workers,config)
        self.num_of_batches = ray.get(self.workers[0].num_of_batches.remote())
        for i in range(len(self.workers)):
            x = ray.get(self.workers[i].addSupervisor.remote(self.supervisor))
            y = ray.get(self.workers[i].addFriend.remote(self.workers[(i-1)%(len(self.workers))]))
        updateWeightsAndBiases = ray.get([self.workers[id+1].receiveParams.remote() for id in range(len(self.workers)-1)])

    


        

    def train(self, circular = True):
        if circular:
            for epoch in range(self.epochs):
                for batch_no in range(int(self.num_of_batches/len(self.workers))):
                    if batch_no%5==0:
                        print(2*batch_no, ' processed!')
                    a = ray.get(self.supervisor.subworkCircularCommunication.remote(batch_no))
                    x = ray.get([self.workers[i].receiveGradientsCircularCommunication.remote() for i in range(len(self.workers))])
                    b = ray.get(self.supervisor.subworkUpdateParameters.remote(self.learning_rate))
        else:
            for epoch in range(self.epochs):
                for batch_no in range(int(self.num_of_batches/len(self.workers))):
                    if batch_no%5==0:
                        print(2*batch_no, ' processed!')
                    a = ray.get(self.supervisor.subworkLinearCommunication.remote(batch_no))
                    x = ray.get([w.receiveGradientsLinearCommunication.remote() for w in self.workers])
                    b = ray.get(self.supervisor.subworkUpdateParameters.remote(self.learning_rate))
                
    def predict(self):
        predict = []
        for w in self.workers:
            predict.append(ray.get(w.predict.remote()))
        return predict



    

        


def distributedTraining(config_filename):
    head = DistributedBolt(['3','11'], config_filename) 
    head.train(False)
    return head.predict()



    