import ray
import os
import toml
import subprocess
from .Worker import Worker
from .Supervisor import Supervisor
import time as time
from .util import initLogging

class DistributedBolt:
    def __init__(self, worker_nodes, config_filename):
        logging = initLogging()
        
        os.system("pip3 install --no-cache-dir ray[default]")
        os.system("export PATH=$PATH:/home/$USER/.local/bin")
        os.system("ray stop")
        os.system('ray up setup.yaml')
        
        logging.info('Ray set up on head node completed.')
        logging.info('Starting the cluster Setup.')

        subprocess.run(["sh", "make_cluster.sh", " ".join(worker_nodes)])

        logging.info('Cluster has started.')
        self.no_of_workers = len(worker_nodes)+1
        current_working_directory = os.getcwd()
        runtime_env = {"working_dir": current_working_directory, "pip": ["toml", "typing", "typing_extensions", 'psutil'], "env_vars": {"OMP_NUM_THREADS": "100"}}
        ray.init(address='auto', runtime_env=runtime_env)
        config = toml.load(config_filename)
        self.epochs = config["params"]["epochs"]
        self.learning_rate = config["params"]["learning_rate"]
        self.layers = [config["dataset"]["input_dim"]]
        for i in range(len(config['layers'])):
            self.layers.append(config['layers'][i]['dim'])
        random_seed = int(time.time())%int(0xffffffff)
        self.workers = [Worker.options(max_concurrency=2).remote(self.layers,config, id+1, self.no_of_workers, id) for id in range(self.no_of_workers)]
        self.supervisor = Supervisor.remote(self.layers,self.workers,config, logging)
        self.num_of_batches = ray.get(self.workers[0].num_of_batches.remote())
        for i in range(len(self.workers)):
            x = ray.get(self.workers[i].addSupervisor.remote(self.supervisor))
            y = ray.get(self.workers[i].addFriend.remote(self.workers[(i-1)%(len(self.workers))]))
        updateWeightsAndBiases = ray.get([self.workers[id+1].receiveParams.remote() for id in range(len(self.workers)-1)])
        self.bolt_computation_time = 0
        self.python_computation_time = 0
        self.communication_time = 0

    def train(self, circular = True):
        if circular:
            logging.info('Circular communication pattern is choosen')
        else:
            logging.info('Linear communication pattern is choosen')
        if circular:
            for epoch in range(self.epochs):
                for batch_no in range(int(self.num_of_batches/len(self.workers))):
                    if batch_no%5==0:
                        print(batch_no, ' processed!')
                    a = ray.get(self.supervisor.subworkCircularCommunication.remote(batch_no))
                    x = ray.get([self.workers[i].receiveGradientsCircularCommunication.remote() for i in range(len(self.workers))])
                    b = ray.get(self.supervisor.subworkUpdateParameters.remote(self.learning_rate))
        else:
            for epoch in range(self.epochs):
                for batch_no in range(int(self.num_of_batches/len(self.workers))):
                    if batch_no%5==0:
                        print(batch_no, ' processed!')
                    gradient_computation_time, getting_gradient_time, summing_and_averaging_gradients_time = ray.get(self.supervisor.subworkLinearCommunication.remote(batch_no))
                    start_gradients_send_time = time.time() 
                    x = ray.get([w.receiveGradientsLinearCommunication.remote() for w in self.workers])
                    gradient_send_time = time.time() - start_gradients_send_time
                    start_update_parameters_time = time.time()
                    b = ray.get(self.supervisor.subworkUpdateParameters.remote(self.learning_rate))
                    update_parameters_time = time.time() - start_update_parameters_time
                    self.bolt_computation_time += gradient_computation_time + update_parameters_time
                    self.python_computation_time += summing_and_averaging_gradients_start_time
                    self.communication_time += getting_gradient_time + gradient_send_time
                    print(self.bolt_computation_time, self.python_computation_time, self.communication_time)
                logging.info('Epoch No:{i}, Bolt Computation Time:{k} Python Computation Time:{l} Communication Time:{m}',epoch,self.bolt_computation_time, self.python_computation_time, self.communication_time)
                
    def predict(self):
        predict = []
        for w in self.workers:
            predict.append(ray.get(w.predict.remote()))
        return predict

    

