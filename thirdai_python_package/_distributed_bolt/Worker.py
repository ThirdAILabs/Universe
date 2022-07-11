from thirdai._thirdai import bolt, dataset
import numpy as np
import ray
from .utils import create_fully_connected_layer_configs, load_dataset
import time


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
        # t1 = time.time()
        self.network.calculateGradientSingleNode(batch_no, self.loss)
        # print('Calcualate Gradient Time: %lf', time.time() - t1)
        return True
    
    def getCalculatedGradients(self):
        w_gradients = []
        b_gradients = []
        t1 = time.time()
        for layer in range(len(self.layers)-1):
            x = self.network.get_weights_gradients(layer)
            y = self.network.get_biases_gradients(layer)
            w_gradients.append(x)
            b_gradients.append(y)
        getting_gradient_time = time.time() - getting_gradient_start_time
        return (w_gradients, b_gradients)

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
    