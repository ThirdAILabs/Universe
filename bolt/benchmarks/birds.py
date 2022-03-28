from thirdai import bolt, dataset
import sys
import argparse
from helpers import add_arguments
import requests
import time
from datetime import date
import numpy as np

# Add the logging folder to the system path
import sys

sys.path.insert(1, sys.path[0] + "/../../logging/")
from mlflow_logger import ExperimentLogger

def _define_network(args):
    layers = [
        bolt.LayerConfig(dim=3000, activation_function=bolt.ActivationFunctions.ReLU, load_factor=args.sparsity, 
            sampling_config=bolt.SamplingConfig(hashes_per_table=args.hashes_per_table, num_tables=args.num_tables, range_pow=14, reservoir_size=32)),
        bolt.LayerConfig(dim=100, activation_function=bolt.ActivationFunctions.Softmax)
        ]

    network = bolt.Network(layers=layers, input_dim=1536)
    return network

def train_birds(args, network, mlflow_logger):
    tr_emb = np.load('/media/scratch/data/birds/extracted/tr_emb1.npy')
    tr_labels = np.load('/media/scratch/data/birds/extracted/tr_labels.npy')

    tst_emb = np.load('/media/scratch/data/birds/extracted/tst_emb.npy')
    tst_labels = np.load('/media/scratch/data/birds/extracted/tst_labels.npy')

    mlflow_logger.log_start_training()
    for _ in range(100):
        network.train(tr_emb, tr_labels, batch_size=32768, loss_fn=bolt.CategoricalCrossEntropyLoss(), learning_rate=args.lr, epochs=1, rehash=3000, rebuild=10000)
        acc, __ = network.predict(tst_emb, tst_labels, batch_size=2048, metrics=["categorical_accuracy"], verbose=False)
        mlflow_logger.log_epoch(acc["categorical_accuracy"][0]
)
    
    final_accuracy = network.predict(tst_emb, tst_labels, batch_size=2048)
    mlflow_logger.log_final_accuracy(final_accuracy)



def main():

    parser = argparse.ArgumentParser(
        description=f"Run BOLT on Amazon 670k with specified params."
    )

    args = add_arguments(
        parser=parser,
        train="/share/data/birds/train.svm",
        test="/share/data/birds/test.svm",
        epochs=25,
        hashes_per_table=4,
        num_tables=64,
        sparsity=0.05,
        lr=0.0001,
    )

    layers = [
        bolt.LayerConfig(dim=3000, activation_function=bolt.ActivationFunctions.ReLU, load_factor=0.05, 
            sampling_config=bolt.SamplingConfig(hashes_per_table=4, num_tables=64, range_pow=14, reservoir_size=32)),
        bolt.LayerConfig(dim=100, activation_function=bolt.ActivationFunctions.Softmax)
        ]

    network = bolt.Network(layers=layers, input_dim=1536)

    network = _define_network(args)

    with ExperimentLogger(
        experiment_name="Birds Benchmark",
        dataset="birds",
        algorithm="Bolt",
    ) as mlflow_logger:
        train_birds(args, network, mlflow_logger)


if __name__ == "__main__":
    main()
