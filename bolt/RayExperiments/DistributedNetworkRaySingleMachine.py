from thirdai import bolt, dataset
import numpy as np
import ray
import os
from ray.util import inspect_serializability


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
    train_x, train_y = dataset.load_bolt_svm_dataset("mnist", 250)
    test_x, test_y = dataset.load_bolt_svm_dataset("mnist.t", 250)
    return train_x, train_y, test_x, test_y


@ray.remote(num_cpus=10)
class DistributedNetworkRay:
    def __init__(self):
        layers = [
            bolt.FullyConnected(dim=256, activation_function="ReLU"),
            bolt.FullyConnected(
                dim=10,
                activation_function="Softmax",
            ),
        ]
        self.network = bolt.DistributedNetwork(layers=layers, input_dim=784)
        self.epochs = 10
        self.train_data, self.train_label, self.test_data, self.test_label = load_mnist()
        self.learning_rate = 0.0005


    def train(self):
        self.batch_size = self.network.initTrainDistributed(
        self.train_data, 
        self.train_label,
        rehash=3000,
        rebuild=10000,
        verbose=True,
        batch_size=64,)
        print(os.environ['OMP_NUM_THREADS'])
        for epoch in range(self.epochs):
            print('Epoch: ', epoch)
            for batch_no in range(self.batch_size):
                self.network.calculateGradientDistributed(batch_no, bolt.CategoricalCrossEntropyLoss())
                self.network.updateParametersDistributed(self.learning_rate)

    def test(self):
        acc, _ = self.network.predictDistributed(
            self.test_data, self.test_label, self.batch_size, ["categorical_accuracy"], verbose=False
        )
        return acc["categorical_accuracy"]

if __name__ == "__main__":
    ray.init(runtime_env={
                "env_vars": {
                    "OMP_NUM_THREADS": "4"}
            })
    setup_module()
    distributed_network = DistributedNetworkRay.remote()
    distributed_network.train.remote()
    acc = ray.get(distributed_network.test.remote())
    print(acc)