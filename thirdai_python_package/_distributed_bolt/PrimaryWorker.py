import numpy as np
import ray
import time
from typing import Tuple, Any, Optional, Dict, List
from .utils import initLogging
from .Worker import Worker



@ray.remote(num_cpus=48, max_restarts=2)
class PrimaryWorker(Worker):
    """
        This is a ray remote class(Actor). Read about them here. 
        (https://docs.ray.io/en/latest/ray-core/actors.html)

        PrimaryWorker is a ray actor which inherits all the function from 
        Worker class. Apart from acting as a Worker, it also extends the worker 
        class to implement functions to control the training. It controls 
        training on each of the node(which batch number to train) and communication
        between the worker nodes.
        
        
        Args:
            layers: List of layer dimensions.
            config: configuration file
            workers: number of workers in training
    """
    def __init__(
        self, 
        layers: List, 
        config, 
        no_of_workers,

    ):
        self.layers = layers
        super().__init__(self.layers, config, no_of_workers, 0)
    
    def addWorkers(
        self,
        workers: List
    ):
        self.workers = workers

    def subworkCircularCommunication(
        self, 
        batch_no: int
    ):
        """

            This function first call the workers to compute the gradients on their network 
            and then implements Baidu's All Ring All Reduce algorithm for communication.
            Read more about that here: 
            https://andrew.gibiansky.com/blog/machine-learning/baidu-allreduce/.
            


            Args:
                batch_no: batch number for the particular worker with worker id (id).
        """
        communication_time = 0
        averaging_gradients_time = 0
        gradient_computation_time = 0

        t1 = time.time()
        updates = ray.get([self.workers[id].calculateGradientsCircular.remote(batch_no) for id in range(len(self.workers))])
        gradient_computation_time += time.time() - t1
        # First Run
        # mt = time.time()
        update_id = 0
        for i in range(self.total_nodes-1):
            if i == self.total_nodes - 2:
                blocking_run = ray.get([w.processRing.remote(update_id, avg_gradients=True) for w in self.workers])
                averaging_gradients_time += sum(i for i,j in blocking_run)/self.total_nodes
                communication_time += sum(j for i,j in blocking_run)/self.total_nodes
            else:
                blocking_run = ray.get([w.processRing.remote(update_id) for w in self.workers])
                averaging_gradients_time += sum(i for i,j in blocking_run)/self.total_nodes
                communication_time += sum(j for i,j in blocking_run)/self.total_nodes
            update_id -= 1
        # print('AllReduce Time:', time.time() - mt)
        # Second Run 
        # mt = time.time()
        update_id = 1
        for i in range(self.total_nodes-1):
            blocking_run = ray.get([w.processRing.remote(update_id, reduce=False) for w in self.workers])
            averaging_gradients_time += sum(i for i,j in blocking_run)/self.total_nodes
            communication_time += sum(j for i,j in blocking_run)/self.total_nodes
            update_id -= 1
        
        # print('AllGather Time:', time.time() - mt)
        return gradient_computation_time, communication_time, averaging_gradients_time


    def subworkLinearCommunication(
        self, 
        batch_no: int
    ):
        """
            This function implements the linear way of communicating between the node.
            In this way of communication, each of the worker calculates their gradients, 
            send their gradients to the supervisor and the supervisor sums the gradients, 
            averages it and and send the gradients back to the workers.



            Args:
                batch_no: batch number for the particular worker with worker id (id).
        """
        start_gradient_computation = time.time()
        calculateGradients = ray.get([self.workers[id].calculateGradientsLinear.remote(batch_no) for id in range(len(self.workers))])
        gradient_computation_time = time.time() - start_gradient_computation
        start_getting_gradients = time.time()
        gradients_list = ray.get([self.workers[id].getCalculatedGradients.remote() for id in range(len(self.workers))])
        getting_gradient_time = time.time() - start_getting_gradients
        
        summing_and_averaging_gradients_start_time = time.time()
        
        self.w_gradients_avg = np.array([np.zeros((self.layers[layer_no+1], self.layers[layer_no])) for layer_no in range(len(self.layers)-1)])
        self.b_gradients_avg = np.array([np.zeros((self.layers[layer_no+1])) for layer_no in range(len(self.layers)-1)])
        
        node_id = 0
        for w_gradients,b_gradients in gradients_list:
            self.w_gradients_avg += w_gradients
            self.b_gradients_avg += b_gradients
            node_id+=1
        
        
        self.w_gradients_avg = np.divide(self.w_gradients_avg, len(self.workers))
        self.b_gradients_avg = np.divide(self.b_gradients_avg, len(self.workers))
        
        summing_and_averaging_gradients_time = time.time() - summing_and_averaging_gradients_start_time
        return gradient_computation_time, getting_gradient_time, summing_and_averaging_gradients_time

    
    def gradients_avg(
        self
    ):
        """
            This function is called by the workers to get the gradients back from PrimaryWorker.
            Calling this function returns the averaged gradients which is already calculated 
            by the PrimaryWorker.
        """
        return self.w_gradients_avg, self.b_gradients_avg

    
    def subworkUpdateParameters(
        self,
        learning_rate: float
    ):
        """

            This function calls every worker to update their parameters(weight and biases) with the
            updated gradients(which they receive from the PrimaryWorker)

            Args:
                learning_rate: learning_rate for the training
        """
        check_update_parameter = ray.get([w.updateParameters.remote(learning_rate) for w in self.workers])
        return True

    
    def check_weights(
        self
    ):
        """
            This is a debug function to see whether the parameters are set accurately or not.
        """
        weights_0, biases_0 = ray.get(self.workers[0].returnParams.remote())
        weights_1, biases_1 = ray.get(self.workers[1].returnParams.remote())


    def weights_biases(
        self
    ):
        """

            This function is called by all the workers(other than worker with id = 0), here 
            all the workers get the same initialized weights and bias as that of worker with id 0 
        """
        
        print('Updating weights & bias parameters across nodes')

        self.weights_biases = ray.get(self.workers[0].returnParams.remote())
        return self.weights_biases
