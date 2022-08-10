from thirdai._thirdai import bolt, dataset
import numpy as np
import ray
from .utils import create_fully_connected_layer_configs, load_dataset
import time
from typing import Tuple, Any, Optional, Dict, List
import time

@ray.remote(num_cpus=20, max_restarts=1)
class Worker:
    """
        This is a ray remote class(Actor). Read about them here. 
        (https://docs.ray.io/en/latest/ray-core/actors.html)
        
        Worker is a ray actor which implements all the lower level 
        functionalities between the Distributed Bolt APIs and
        Bolt native code.

        Args:
            layers: List of layer dimensions
            config: configuration file for setting up the network
            total_nodes: total number of nodes
            id: id of this particular worker
    """

    def __init__(self, 
        layers: List, 
        config, 
        total_nodes: int, 
        id: int
    ):
        print('Worker Started')
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
        if len(config["dataset"]["train_data"]) != total_nodes:
            raise ValueError("Give n trainging examples for n nodes.")

        data = load_dataset(config, total_nodes, id)
        if data is None:
            raise ValueError("Unable to load a dataset. Please check the config")

        if config["params"]["loss_fn"].lower() == "categoricalcrossentropyloss":
            self.loss = bolt.CategoricalCrossEntropyLoss()
        elif config["params"]["loss_fn"].lower() == "meansquarederror":
            self.loss = bolt.MeanSquaredError()
        else:
            print("'{}' is not a valid loss function".format(config["params"]["loss_fn"]))

        
        self.train_data, self.train_label, self.test_data, self.test_label = data
        
        self.num_of_batches = self.network.initTrainSingleNode(
                    self.train_data, 
                    self.train_label,
                    rehash=self.rehash,
                    rebuild=self.rebuild,
                    verbose=False)
        
        
        self.total_nodes = total_nodes
        self.id = id


    def addSupervisor(
        self, 
        supervisor
    ):
        """

            This function assigns each of the worker their supervisor
        """
        self.supervisor = supervisor
        

    def addFriend(
        self, 
        friend
    ):
        """

            This function is only needed for circular way of communication.
            This function assigns each of the worker their friend to which
            they will be communicating their gradients. Look at this link:
            https://andrew.gibiansky.com/blog/machine-learning/baidu-allreduce/ 
        """
        self.friend = friend

    def calculateGradientsCircular(
        self, 
        batch_no: int
    ):
        """

            This function is called only when the mode of 
            communication is circular.


            This functions calls the API 'calculateGradientSingleNode',
            which calculates the gradients for the network managed by
            this particular worker. The calculateGradientSingleNode
            calculates the gradient for the particular training batch 
            with batch no. batch_no and with loss function specified
            in the config.

            This function also defines the partition size which defines the
            size of block of gradients which are communicated between a worker 
            and its friend.

            Args:
                batch_no: training batch to calculate gradients on.
 
        """
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

    def calculateGradientsLinear(self, 
        batch_no: int,
        compression=None,
        compression_density=0.1
    ):
        """
            This function is called only when the mode of communication is
            linear.

            This functions calls the API 'calculateGradientSingleNode',
            which calculates the gradients for the network managed by
            this particular worker. The calculateGradientSingleNode
            calculates the gradient for the particular training batch 
            with batch no. batch_no and with loss function specified
            in the config.

            Args:
                batch_no: training batch to calculate gradients on.
        """
        self.network.calculateGradientSingleNode(batch_no, self.loss)
        if compression=="DRAGON":
            self.w_sparse_grad,self.b_sparse_grad= self.getDragonGradients(compression_density=compression_density)
            # print(f"worker id {self.id} after calculate gradients {self.w_sparse_grad}")
        return True
    
    def approximate_topk(self,weights,top_frac):
        n=int(top_frac*weights.shape[0])
        vals=np.partition(weights,n)[-n:]
        return np.min(vals)
    
    
    def getDragonGradients(self,compression_density):
        # print(f"calculating the dragon gradients for the workerid {self.id}")
        start=time.time()
        w_sparse_grad=[]
        b_sparse_grad=[]

        for layer in range(len(self.layers)-1):
            x = self.network.get_weights_gradients(layer)
            y = self.network.get_biases_gradients(layer)
            x=np.ravel(x)
            y=np.ravel(y)

            m_x=int(compression_density*x.shape[0])
            m_y=int(compression_density*y.shape[0])
            thresh_x=0
            thresh_y=0

            num_samples=2
            for i in range(num_samples):

                sampled_x=np.random.choice(x.shape[0],min(x.shape[0],100000))
                sampled_y=np.random.choice(y.shape[0],min(y.shape[0],100000))

                thresh_x+=self.approximate_topk(np.abs(x[sampled_x]),compression_density)/num_samples
                thresh_y+=self.approximate_topk(np.abs(y[sampled_y]),compression_density)/num_samples

        
            idx=np.where((x>thresh_x) | (x<-1*thresh_x))[0].astype(int)
            idy=np.where((y>thresh_y) | (y<-1*thresh_y))[0].astype(int)

            indices_x=idx[np.random.choice(idx.shape[0],min(idx.shape[0],m_x))]
            indices_y=idy[np.random.choice(idy.shape[0],min(idy.shape[0],m_y))]

            # np.random.shuffle(indices_x)
            # np.random.shuffle(indices_y)

            # indices_x=indices_x[:m_x]
            # indices_y=indices_y[:m_y]

            vals_x=x[indices_x]
            vals_y=y[indices_y]

            w_sparse_grad.append((indices_x,vals_x))
            b_sparse_grad.append((indices_y,vals_y))
        
        return (w_sparse_grad,b_sparse_grad)



    def getCalculatedGradients(
        self,
        compression=None,compression_density=0.1
    ):
        """
            This function is called only when the mode of communication
            is Linear.

            This function is called by the supervisor to compute the 
            averages of the calculated gradients. This functions
            calls 'get_weights_gradient' and 'get_biases_gradients' functions
            inside bolt to take the gradients and return them to supervisor.
        """
        if compression is not None and compression =="DRAGON":
            return self.w_sparse_grad,self.b_sparse_grad

        w_gradients = []
        b_gradients = []
        for layer in range(len(self.layers)-1):
            x = self.network.get_weights_gradients(layer)
            y = self.network.get_biases_gradients(layer)
            w_gradients.append(x)
            b_gradients.append(y)
        return (w_gradients, b_gradients)

    def returnParams(
        self
    ):
        """

            This function will only be called for worker having its id 0.
            The supervisor will call this function to get the initial random 
            weights from worker with id 0 and then send those weights to all
            the workers.

        """
        weights = []
        biases = []
        for layer in range(len(self.layers)-1):
            x = self.network.get_weights(layer)
            y = self.network.get_biases(layer)
            weights.append(x)
            biases.append(y)
        return weights, biases

    def receiveParams(
        self
    ):
        """

            This function is called by supervisor to all the workers whose id 
            is not equal to 0. This function gets the initialized random weight
            ans biases from worker with id = 0. and sets the weight on all
            the other workers.

        """
        weights, biases = ray.get(self.supervisor.weights_biases.remote())
        for layer in range(len(weights)):
            self.network.set_weights(layer, weights[layer])
            self.network.set_biases(layer, biases[layer])
        return True

    def receiveGradientsCircularCommunication(
        self
    ):
        """

            This function is called only when the communication pattern choosen
            is circular.

            This function is called by the supervisor to make set the updated 
            gradients to the network.

        """
        for layer in range(len(self.w_gradients)):
            self.network.set_weights_gradients(layer, self.w_gradients[layer])
            self.network.set_biases_gradients(layer, self.b_gradients[layer])
        return True


    def receiveDragonGradients(self):

        num_workers=ray.get(self.supervisor.num_workers.remote())

        w_sparse_grads,b_sparse_grads=ray.get(self.supervisor.sparse_grads.remote())
        for layer in range(len(self.layers)-1):
            
            shape=(self.layers[layer],self.layers[layer+1])
            w_gradient=np.zeros(shape[0]*shape[1])
            b_gradient=np.zeros(shape[1])
            
            for node_weights in w_sparse_grads:
                np.add.at(w_gradient,node_weights[layer][0],node_weights[layer][1])
            for node_biases in b_sparse_grads:
                np.add.at(b_gradient,node_biases[layer][0],node_biases[layer][1])
            
            self.network.set_weights_gradients(layer, w_gradient.reshape(shape[1],shape[0])/num_workers)
            self.network.set_biases_gradients(layer, b_gradient/num_workers)
        
        return True


    def receiveGradientsLinearCommunication(
        self,
        compression=None
    ):
        """

            This function is called only when the communication pattern choosen
            is linear.

            This function is called by the supervisor to first, get the updated gradients 
            from the supervisor and then set those updated gradients to the network.
            
        """

        if compression=="DRAGON":
            self.receiveDragonGradients()
            return True
        
        w_gradients_updated, b_gradients_updated = ray.get(self.supervisor.gradients_avg.remote())
        for layer in range(len(w_gradients_updated)):
            self.network.set_weights_gradients(layer, w_gradients_updated[layer])
            self.network.set_biases_gradients(layer, b_gradients_updated[layer])
        return True
    
    def processRing(
        self,
        update_id: int, 
        reduce: Optional[bool] = True, 
        avg_gradients: Optional[bool] = False):
        """

            This function contains the main code for the circular ring communication
            pattern. 
            
            The function first calculates the partition index range on which it will 
            work, then get the graidnets on that range from its friend worker and sums
            it to the partition the partition the current worker.

            Here Each of the node communicates the partitioned gradients with 
            their friend nodes, and those friend node communicate with their friends 
            and the communication there by happens in a circle.  


            Args:
                update_id: This id is use to calculate the partition to work on.
                reduce: This bool determines whether we need to reduce or gather
                    True: redue, Flase: Gather
                avg_gradients: This bool determines whether we will average the 
                    gradients, or not. True: do averaging(i.e., divide by total_nodes), 
                    False: Do nothing

        """
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

    

    def receiveArrayPartitions(self, 
        update_id: int
    ):
        """
            This function will only be get called for circular ring communication
            pattern.

            This function returns the array partition to the worker it is called by. 


            Args:
                update_id: This id is use to calculate the partition to work on.
        """
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

    

    def updateParameters(
        self, 
        learning_rate: float
    ):
        """

            This function calls updateParameter function inside bolt, which
            inherently updates the entire network.

            Args:
                learning_rate: the learning rate for updating the parameters
        """
        self.network.updateParametersSingleNode(learning_rate)
        return True
        
    
    def num_of_batches(
        self
    ):
        """

            This function returns the total number of batches the workers have.
        """
        return self.num_of_batches

    def predict(
        self
    ):
        """
            This function calls the predict function(predictSingleNode) to return the 
            prediction from the network manges by this single worker.
        """
        acc= self.network.predictSingleNode(
            self.test_data, self.test_label, self.num_of_batches, False, ["categorical_accuracy"], verbose=False
        )
        return acc
    