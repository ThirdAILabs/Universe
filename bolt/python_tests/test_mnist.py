import os
from thirdai import bolt, dataset
import sys
from helpers import train
import pytest
from collections import namedtuple


def setup_module():
	if not os.path.exists("mnist"):
		os.system("curl https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.bz2 --output mnist.bz2")
		os.system("bzip2 -d mnist.bz2")

	if not os.path.exists("mnist.t"):
		os.system("curl https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.t.bz2 --output mnist.t.bz2")
		os.system("bzip2 -d mnist.t.bz2")



def train_mnist_sparse_output_layer(args):
	layers = [
		bolt.LayerConfig(dim=256, activation_function="ReLU"),
		bolt.LayerConfig(dim=10, load_factor=args.sparsity,
											activation_function="Softmax",
											sampling_config=bolt.SamplingConfig(
													hashes_per_table=args.hashes_per_table, num_tables=args.num_tables,
													range_pow=args.hashes_per_table * 3, reservoir_size=10))
	]
	network = bolt.Network(layers=layers, input_dim=784)

	train_data = dataset.loadSVMDataset(args.train, 250)
	test_data = dataset.loadSVMDataset(args.test, 250)
	epoch_times = []
	epoch_accuracies = []
	for _ in range(args.epochs):
		times = network.train(train_data, args.lr, 1,
													rehash=3000, rebuild=10000)
		epoch_times.append(times[0])
		acc = network.predict(test_data)
		epoch_accuracies.append(acc)

	return epoch_accuracies[-1], epoch_accuracies, epoch_times


def train_mnist_sparse_hidden_layer(args):
	layers = [
		bolt.LayerConfig(dim=20000, load_factor=args.sparsity,
											activation_function="ReLU",
											sampling_config=bolt.SamplingConfig(
													hashes_per_table=args.hashes_per_table, num_tables=args.num_tables,
													range_pow=args.hashes_per_table * 3, reservoir_size=32)),
		bolt.LayerConfig(dim=10, activation_function="Softmax")
	]
	network = bolt.Network(layers=layers, input_dim=784)

	train_data = dataset.loadSVMDataset(args.train, 250)
	test_data = dataset.loadSVMDataset(args.test, 250)
	epoch_times = []
	epoch_accuracies = []
	for _ in range(args.epochs):
		times = network.train(train_data, args.lr, 1,
													rehash=3000, rebuild=10000)
		epoch_times.append(times[0])
		acc = network.predict(test_data)
		epoch_accuracies.append(acc)
	return epoch_accuracies[-1], epoch_accuracies, epoch_times


@pytest.mark.integration
def test_mnist_sparse_output():
	args = {
		"dataset": "mnist",
		"train": "mnist",
		"test": "mnist.t",
		"epochs": 10,
		"hashes_per_table": 1,
		"num_tables": 32,
		"sparsity": 0.4,
		"lr": 0.0001,
		"enable_checks": True,
		"runs": 1
	}
	# Turn the dictionary into the format expected by the train method, fields
	# referencable by e.g. args.train
	args = namedtuple("args", args.keys())(*args.values())
	accs, times = train(
		args, train_fn=train_mnist_sparse_output_layer, accuracy_threshold=0.95)

@pytest.mark.integration
def test_mnist_sparse_hidden():
	args = {
		"dataset": "mnist",
		"train": "mnist",
		"test": "mnist.t",
		"epochs": 10,
		"hashes_per_table": 3,
		"num_tables": 64,
		"sparsity": 0.01,
		"lr": 0.0001,
		"enable_checks": True,
		"runs": 1
	}
	args = namedtuple("args", args.keys())(*args.values())
	accs, times = train(
			args, train_fn=train_mnist_sparse_hidden_layer, accuracy_threshold=0.95)