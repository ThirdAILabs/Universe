from thirdai import bolt, dataset
import numpy as np
import sys
import mlflow
import os

sys.path.insert(1, sys.path[0] + "/../benchmarks/")
from mlflow_logger import *

class CookieMonster():
    def __init__(self, fcn_configs):
        if len(fcn_configs) != 2:  
            raise ValueError("Cookie Monster needs two FCNs")
        self.fcn_configs = fcn_configs
        self.bolt_classifier = bolt.TextClassifier(self.fcn_configs)
        self.file_dir = os.path.dirname(os.path.abspath(__file__))
        self.file_name = os.path.join(self.file_dir, "../benchmarks/config.toml")
        with open(self.file_name) as f:
            parsed_config = toml.load(f)
        mlflow.set_tracking_uri(parsed_config["tracking"]["uri"])
        mlflow.set_experiment("Cookie Monster")
        # TODO(henry): would be nice to know the output dimension as a member variable
    
    def download_hidden_weights(self, path_to_weights):
        #TODO(henry): download weights from mlflow and load them into cookie monster
        pass

    def set_output_dimension(self, dimension):
        weights = self.bolt_classifier.get_hidden_weights()
        biases = self.bolt_classifier.get_hidden_biases()
        
        # np.save("weights.npy", weights)

        del(self.bolt_classifier) # delete old model
        new_configs = [self.fcn_configs[0], bolt.FullyConnected(dem=dimension, activation_function="Softmax")]
        self.fcn_configs = new_configs
        self.bolt_classifier = bolt.TextClassifier(self.fcn_configs, n_classes=dimension)

        self.bolt_classifier.set_hidden_weights(weights)
        self.bolt_classifier.set_hidden_biases(biases)

    
    def train_corpus(self, path_to_config_directory):
        mlflow.start_run(
            run_name="train_run"
        )

        #TODO(henry): work out a way to feed data into the model
        rootdir = path_to_config_directory
        for subdir, dirs, files in os.walk(rootdir):
            for file in files:
                path = os.path.join(subdir, file)
                mlflow.log_artifact(path)

                with open(path, "r") as f:
                    config = toml.load(f)
                    train_file = config["train_file"]
                    num_classes = config["num_classes"]
                    if num_classes != self.out_dim:
                        self.set_output_dimension(num_classes)
                    epochs = config["epochs"]
                    learning_rate = config["learning_rate"]

                    self.bolt_classifier.train(train_file=train_file, epochs=epochs, learning_rate=learning_rate)


        mlflow.end_run()
    
    def evaluate(self, path_to_config_directory):
        mlflow.log_artifact("weights.npy")
        # print("Logged weights")

    #TODO(henry): work out a way to manage the versions of weights
    