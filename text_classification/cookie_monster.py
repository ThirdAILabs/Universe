from thirdai import bolt, dataset
import numpy as np
import sys
import mlflow
import os
import toml

sys.path.insert(1, sys.path[0] + "/../benchmarks/")
from mlflow_logger import *


class CookieMonster:
    def __init__(
        self,
        input_dimension,
        hidden_dimension=2000,
        hidden_sparsity=0.1,
        # hidden_sampling_config=None,
    ):
        self.input_dimension = input_dimension
        self.hidden_dim = hidden_dimension
        self.hidden_sparsity = hidden_sparsity
        self.construct(2)

        self.file_dir = os.path.dirname(os.path.abspath(__file__))
        self.file_name = os.path.join(self.file_dir, "../benchmarks/config.toml")
        with open(self.file_name) as f:
            parsed_config = toml.load(f)
        mlflow.set_tracking_uri(parsed_config["tracking"]["uri"])
        mlflow.set_experiment("Cookie Monster")

    def construct(self, output_dim):
        self.input_layer = bolt.graph.Input(dim=self.input_dimension)
        self.hidden_layer = bolt.graph.FullyConnected(
            dim=self.hidden_dim,
            sparsity=self.hidden_sparsity,
            activation="relu",
            # sampling_config=hidden_sampling_config
        )(self.input_layer)
        self.output_layer = bolt.graph.FullyConnected(
            dim=output_dim, activation="softmax"
        )(self.hidden_layer)
        self.model = bolt.graph.Model(
            inputs=[self.input_layer], output=self.output_layer
        )
        self.model.compile(loss=bolt.CategoricalCrossEntropyLoss())

    def set_output_dimension(self, dimension):
        # TODO: write DAG get_dim() python binding
        if self.output_layer.get_dim() == dimension:
            return
        # del self.model
        save_loc = "./hidden_layer_parameters"
        self.hidden_layer.save_parameters(save_loc)

        self.construct(dimension)
        self.hidden_layer.load_parameters(save_loc)

    def train_corpus(
        self,
        path_to_config_directory,
        mlflow_enabled=True,
        evaluate=False,
        verbose=False,
    ):
        if mlflow_enabled and evaluate:
            mlflow.start_run(run_name="evaluation_run")

        if mlflow_enabled and (not evaluate):
            mlflow.start_run(run_name="train_run")

        rootdir = path_to_config_directory
        for subdir, dirs, files in os.walk(rootdir):
            for file in files:
                path = os.path.join(subdir, file)
                if mlflow_enabled:
                    mlflow.log_artifact(path)

                with open(path, "r") as f:
                    print("Training: ", path)
                    config = toml.load(f)
                    train_file = config["train_file"]
                    test_file = config["test_file"]
                    num_classes = config["num_classes"]
                    batch_size = config["batch_size"]
                    if num_classes != self.output_layer.get_dim():
                        raise ValueError("Output dimension is incorrect")

                    self.set_output_dimension(num_classes)
                    epochs = config["epochs"]
                    learning_rate = config["learning_rate"]

                    train_x, train_y = dataset.load_bolt_svm_dataset(
                        train_file, batch_size
                    )
                    test_x, test_y = dataset.load_bolt_svm_dataset(
                        test_file, batch_size
                    )
                    for _ in range(epochs):
                        self.model.train(
                            train_x, train_y, learning_rate=learning_rate, epochs=1
                        )
                        if verbose:
                            metrics = self.model.predict(test_x, test_y)
                            print(
                                "Epoch: ",
                                _,
                                " Accuracy: ",
                                metrics["categorical_accuracy"],
                            )

                    metrics = self.model.predict(test_x, test_y)
                    print("Epoch: ", _, " Accuracy: ", metrics["categorical_accuracy"])

                print("\n")

        if mlflow_enabled:
            save_loc = "./hidden_layer_parameters"
            self.hidden_layer.save_parameters(save_loc)
            mlflow.log_artifact(save_loc)
            mlflow.end_run()

    def evaluate(self, path_to_config_directory):
        self.train_corpus(path_to_config_directory, evaluate=True)

    def download_hidden_parameters(self, link_to_parameter):
        local_param_path = mlflow.artifacts.download_artifacts(link_to_parameter)
        self.hidden_layer.load_parameters(local_param_path)
        print("Loaded parameters")
