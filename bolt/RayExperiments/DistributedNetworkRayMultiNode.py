from thirdai import bolt, dataset
import numpy as np
import ray
import os
import time as time 
import mnist


def setup_module():
    if not os.path.exists("mnist"):
        os.system(
            "curl https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.bz2 --output mnist.bz2"
        )
        os.system("bzip2 -d mnist.bz2")

    if not os.path.exists("mnist.t"):
        os.system(
            "curl https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.t.bz2 --output mnist.t.bz2"
        )
        os.system("bzip2 -d mnist.t.bz2")

def load_mnist():
    train_x, train_y = dataset.load_bolt_svm_dataset("/share/pratik/RayExperiments/mnist", 2000)
    test_x, test_y = dataset.load_bolt_svm_dataset("/share/pratik/RayExperiments/mnist.t", 2000)
    return train_x, train_y, test_x, test_y


@ray.remote(num_cpus=40)
class Worker:
    def __init__(self, layers):
        self.layers = layers
        self.bolt_layers = [
            bolt.FullyConnected(dim=256, activation_function="ReLU"),
            bolt.FullyConnected(
                dim=10,
                activation_function="Softmax",
            ),
        ]
        self.network = bolt.DistributedNetwork(layers=self.bolt_layers, input_dim=784)
        self.train_data, self.train_label, self.test_data, self.test_label = load_mnist()
        self.learning_rate = 0.0005
        self.batch_size = self.network.initTrainDistributed(
                    self.train_data, 
                    self.train_label,
                    rehash=3000,
                    rebuild=10000,
                    verbose=False,)


    def calculateAndReturnGradients(self, batch_no):
        self.network.calculateGradientDistributed(batch_no, bolt.CategoricalCrossEntropyLoss())
        w_gradients = []
        b_gradients = []
        for layer in range(len(self.layers)-1):
            x = self.network.get_weights_gradients(layer)
            y = self.network.get_biases_gradients(layer)
            w_gradients.append(x)
            b_gradients.append(y)
        return w_gradients, b_gradients 


    def receiveGradients(self, layer, w_gradients_updated, b_gradients_updated):
        self.network.set_weights_gradients(layer, w_gradients_updated)
        self.network.set_biases_gradients(layer, b_gradients_updated)
        return True

    def updateParameters(self):
        self.network.updateParametersDistributed(self.learning_rate)
        return True
        
    
    def batch_size(self):
        return self.batch_size

    def predict(self):
        acc, _ = self.network.predictDistributed(
            self.test_data, self.test_label, self.batch_size, ["categorical_accuracy"], verbose=False
        )
        return acc["categorical_accuracy"]
    

@ray.remote(num_cpus=2)
class Supervisor:
    def __init__(self, layers, no_of_workers):
        self.epochs = 2
        self.workers = [Worker.remote(layers) for _ in range(no_of_workers)]
        self.batch_size = ray.get(self.workers[0].batch_size.remote())
        self.layers = layers
        self.gradient_calculcation_time = 0
        self.gradient_distribution_time = 0
        self.updating_parameter_time = 0
    
    def batch_to_train(self, id, batch_no):
        return int(self.batch_size/len(self.workers))*id + batch_no


    def work(self):
        gradients_list = []
        for epoch in range(self.epochs):
            print('Epochs: ', epoch)
            for batch_no in range(int(self.batch_size/len(self.workers))):
                t1 = time.time()
                gradients_list = np.array(ray.get([self.workers[id].calculateAndReturnGradients.remote(self.batch_to_train(id, batch_no)) for id in range(len(self.workers))]))
                t2 = time.time()
                w_gradients_avg = np.array([np.zeros((self.layers[layer_no], self.layers[layer_no+1])) for layer_no in range(len(self.layers)-1)])
                b_gradients_avg = np.array([np.zeros((self.layers[layer_no], self.layers[layer_no+1])) for layer_no in range(len(self.layers)-1)])

                for w_gradients,b_gradients in gradients_list:
                    w_gradients_avg += w_gradients
                    b_gradients_avg += b_gradients
                
                w_gradients_avg = w_gradients_avg/len(self.workers)
                b_gradients_avg = b_gradients_avg/len(self.workers)
        
                return_update_gradients = []
                for layer in range(len(w_gradients_avg)):
                    return_update_gradients = ray.get([w.receiveGradients.remote(layer, w_gradients_avg[layer], b_gradients_avg[layer]) for w in self.workers])
                t3 = time.time()
                check_update_parameter = ray.get([w.updateParameters.remote() for w in self.workers])
                t4 = time.time()
                self.gradient_calculcation_time += t2-t1
                self.gradient_distribution_time += t3-t2
                self.updating_parameter_time += t4-t3
                

        return ray.get(self.workers[0].predict.remote()), ray.get(self.workers[1].predict.remote())
                
    def gettime(self):
        return self.gradient_calculcation_time, self.gradient_distribution_time, self.updating_parameter_time

if __name__=="__main__":
    ray.init(address='auto', runtime_env={
                "env_vars": {
                    "OMP_NUM_THREADS": "100"}
            })
    sup = Supervisor.remote([784,256,10],2)
    print(ray.get(sup.work.remote()))
    print(ray.get(sup.gettime.remote()))


