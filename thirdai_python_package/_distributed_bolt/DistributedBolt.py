import ray
import os
import toml
import subprocess
from .Worker import Worker
from .Supervisor import Supervisor
import time as time

class DistributedBolt:
    def __init__(self, worker_nodes, config_filename):
        os.system("pip3 install --no-cache-dir ray[default]")
        os.system("export PATH=$PATH:/home/$USER/.local/bin")
        os.system("ray stop")
        os.system('ray up setup.yaml')
        subprocess.run(["sh", "make_cluster.sh", " ".join(worker_nodes)])
        self.no_of_workers = len(worker_nodes)+1
        runtime_env = {"working_dir": "/home/pratik/Universe/thirdai_python_package_tests/distributed_training_sample", "pip": ["toml", "typing", "typing_extensions", 'psutil'], "env_vars": {"OMP_NUM_THREADS": "100"}}
        ray.init(address='auto', runtime_env=runtime_env)
        config = toml.load(config_filename)
        self.epochs = config["params"]["epochs"]
        self.learning_rate = config["params"]["learning_rate"]
        self.layers = [config["dataset"]["input_dim"]]
        for i in range(len(config['layers'])):
            self.layers.append(config['layers'][i]['dim'])
        random_seed = int(time.time())%int(0xffffffff)
        self.workers = [Worker.options(max_concurrency=2).remote(self.layers,config, id+1, self.no_of_workers, id) for id in range(self.no_of_workers)]
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
                        print(batch_no, ' processed!')
                    a = ray.get(self.supervisor.subworkCircularCommunication.remote(batch_no))
                    x = ray.get([self.workers[i].receiveGradientsCircularCommunication.remote() for i in range(len(self.workers))])
                    b = ray.get(self.supervisor.subworkUpdateParameters.remote(self.learning_rate))
        else:
            for epoch in range(self.epochs):
                for batch_no in range(int(self.num_of_batches/len(self.workers))):
                    if batch_no%5==0:
                        print(batch_no, ' processed!')
                    a = ray.get(self.supervisor.subworkLinearCommunication.remote(batch_no))
                    x = ray.get([w.receiveGradientsLinearCommunication.remote() for w in self.workers])
                    b = ray.get(self.supervisor.subworkUpdateParameters.remote(self.learning_rate))
                
    def predict(self):
        predict = []
        for w in self.workers:
            predict.append(ray.get(w.predict.remote()))
        return predict

