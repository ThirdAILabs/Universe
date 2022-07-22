import ray
import os
import toml
import textwrap
from .Worker import Worker
from .Supervisor import Supervisor
import time as time
from .utils import initLogging
from typing import Tuple, Any, Optional, Dict, List

class DistributedBolt:
    """
        Implements all the user level Distributed Bolt APIs to the users.
        Args: 
            worker_nodes: Number of workers to start training on.
                This number should be less than equal to the number of nodes(including the head node) training 
                is started.
            config_filename: The name of the config file which is going to be used for training. 
    """
    def __init__(
        self, 
        worker_nodes: int, 
        config_filename: str
        ):

        self.logging = initLogging('main.log')
        self.logging.info('Training has started!')
        
        self.no_of_workers = worker_nodes

        current_working_directory = os.getcwd()
        runtime_env = {"working_dir": current_working_directory, "pip": ["toml", "typing", "typing_extensions", 'psutil'], "env_vars": {"OMP_NUM_THREADS": "100"}}
        
        
        ray.init(address='auto', runtime_env=runtime_env)
        
        if not ray.is_initialized():
            raise Exception(textwrap.dedent("""
                Some issue with cluster setup. Ray is not getting initialized.
                Make sure to have ray cluster online before calling
                Distributed Bolt.
            """))
        
        self.logging.info('Ray Initialized')
        
        config = toml.load(config_filename)

        self.epochs = config["params"]["epochs"]
        self.learning_rate = config["params"]["learning_rate"]
        self.layers = [config["dataset"]["input_dim"]]
        
        for i in range(len(config['layers'])):
            self.layers.append(config['layers'][i]['dim'])
        
        self.workers = [Worker.options(max_concurrency=2).remote(self.layers,config, self.no_of_workers, id) for id in range(self.no_of_workers)]
        self.supervisor = Supervisor.remote(self.layers,self.workers)
        
        self.num_of_batches = min(ray.get([self.workers[i].num_of_batches.remote() for i in range(self.no_of_workers)]))
        
        for i in range(len(self.workers)):
            x = ray.get(self.workers[i].addSupervisor.remote(self.supervisor))
            y = ray.get(self.workers[i].addFriend.remote(self.workers[(i-1)%(len(self.workers))]))
        
        self.bolt_computation_time = 0
        self.python_computation_time = 0
        self.communication_time = 0



    def train(
        self, 
        circular: Optional[bool] = False
        ) -> None:
        """
            Trains the network using the communication type choosen.
            Args:
                circular: True, if circular communication is required.
                        False, if linear ccommunication is required.
        """
        
        if circular:
            self.logging.info('Circular communication pattern is choosen')
            for epoch in range(self.epochs):
                updateWeightsAndBiases = ray.get([self.workers[id+1].receiveParams.remote() for id in range(len(self.workers)-1)])
                for batch_no in range(int(self.num_of_batches/len(self.workers))):
                    if batch_no%5==0:
                        self.logging.info(str(batch_no) + ' processed!, Total Batches: ' + str(self.num_of_batches))
                    a = ray.get(self.supervisor.subworkCircularCommunication.remote(batch_no))
                    x = ray.get([self.workers[i].receiveGradientsCircularCommunication.remote() for i in range(len(self.workers))])
                    b = ray.get(self.supervisor.subworkUpdateParameters.remote(self.learning_rate))
        else:
            self.logging.info('Linear communication pattern is choosen')
            updateWeightsAndBiases = ray.get([self.workers[id+1].receiveParams.remote() for id in range(len(self.workers)-1)])
            for epoch in range(self.epochs):
                for batch_no in range(self.num_of_batches):
                    if batch_no%5==0:
                        self.logging.info(str(batch_no) + ' processed!, Total Batches: ' + str(self.num_of_batches))
                    gradient_computation_time, getting_gradient_time, summing_and_averaging_gradients_time = ray.get(self.supervisor.subworkLinearCommunication.remote(batch_no))
                    start_gradients_send_time = time.time() 
                    x = ray.get([w.receiveGradientsLinearCommunication.remote() for w in self.workers])
                    gradient_send_time = time.time() - start_gradients_send_time
                    start_update_parameters_time = time.time()
                    b = ray.get(self.supervisor.subworkUpdateParameters.remote(self.learning_rate))
                    update_parameters_time = time.time() - start_update_parameters_time
                    self.bolt_computation_time += gradient_computation_time + update_parameters_time
                    self.python_computation_time += summing_and_averaging_gradients_time
                    self.communication_time += getting_gradient_time + gradient_send_time
                self.logging.info('Epoch No: ' + str(epoch) + ', Bolt Computation Time: ' + str(self.bolt_computation_time) + ', Python Computation Time: ' + str(self.python_computation_time) + ', Communication Time: ' + str(self.communication_time))
                for i in range(len(self.workers)):
                    acc, _ = ray.get(self.workers[i].predict.remote())
                    self.logging.info('Accuracy on workers %d: %lf', i, acc["categorical_accuracy"])
                
                
    def predict(
        self
        ):
        """
            Calls network.predict() on one of worker on head node and returns the predictions.
        """  

        assert len(self.workers) > 0, 'No workers are initialized now.'
        return (ray.get(self.workers[0].predict.remote()))

    

