from thirdai import bolt, dataset
import numpy as np
import sys
import mlflow

sys.path.insert(1, sys.path[0] + "/../benchmarks/")
from mlflow_logger import *

class CookieMonster():
    def __init__(self, size="2Gb", output_dimension=2):
        self.model_size = size
        self.bolt_classifier = bolt.TextClassifier(model_size=size, n_classes=output_dimension)
        self.out_dim = output_dimension
    
    def download_hidden_weights(self, path_to_weights):
        # download weights from mlflow and load them into bolt
        pass

    def upload_hidden_weights(self, path_to_weights):
        pass

    def set_output_dimension(self, dimension):
        weights = self.bolt_classifier.get_hidden_weights()
        biases = self.bolt_classifier.get_hidden_biases()
        
        print(weights.shape)
        np.save("weights.npy", weights)

        del(self.bolt_classifier) # delete old model

        self.bolt_classifier = bolt.TextClassifier(model_size=self.model_size, n_classes=dimension)

        self.bolt_classifier.set_hidden_weights(weights)
        self.bolt_classifier.set_hidden_biases(biases)

    
    def train(self):
        file_dir = os.path.dirname(os.path.abspath(__file__))
        file_name = os.path.join(file_dir, "./config.toml")
        with open(file_name) as f:
            parsed_config = toml.load(f)
        mlflow.set_tracking_uri(parsed_config["tracking"]["uri"])

        mlflow.set_experiment("Cookie Monster")
        mlflow.start_run(
            run_name="test_run"
        )

        mlflow.log_artifact("weights.npy")
        print("Logged weights")

        mlflow.end_run()
    