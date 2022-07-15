from thirdai import bolt, dataset
import numpy as np

# Uncomment the following line when used on a machine with valid mlflow credentials
# import mlflow
import os


class CookieMonster:
    def __init__(
        self,
        input_dimension,
        hidden_dimension=2000,
        output_dimension=2,
        hidden_sparsity=0.1,
        mlflow_enabled=True,
    ):
        import toml

        self.input_dimension = input_dimension
        self.hidden_dim = hidden_dimension
        self.hidden_sparsity = hidden_sparsity
        self.mlflow_enabled = mlflow_enabled
        self.construct(output_dimension)

        self.config_file_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_file_name = os.path.join(
            self.config_file_dir, "../benchmarks/config.toml"
        )
        with open(self.config_file_name) as f:
            parsed_config = toml.load(f)

        if self.mlflow_enabled:
            mlflow.set_tracking_uri(parsed_config["tracking"]["uri"])
            mlflow.set_experiment("Cookie Monster")

    def construct(self, output_dim):
        self.input_layer = bolt.graph.Input(dim=self.input_dimension)
        self.hidden_layer = bolt.graph.FullyConnected(
            dim=self.hidden_dim,
            sparsity=self.hidden_sparsity,
            activation="relu",
        )(self.input_layer)
        self.output_layer = bolt.graph.FullyConnected(
            dim=output_dim, activation="softmax"
        )(self.hidden_layer)

        self.model = bolt.graph.Model(
            inputs=[self.input_layer], output=self.output_layer
        )

        self.model.compile(loss=bolt.CategoricalCrossEntropyLoss())

    def set_output_dimension(self, dimension):
        if self.output_layer.get_dim() == dimension:
            return
        save_loc = "./hidden_layer_parameters"
        self.hidden_layer.save_parameters(save_loc)

        self.construct(dimension)
        self.hidden_layer.load_parameters(save_loc)
        os.remove(save_loc)

    def eat_corpus(
        self,
        path_to_config_directory,
        evaluate=False,
        verbose=False,
    ):
        """
        Given a directory containing only .txt config files, this function trains each dataset with the parameters specified in each config file.
        Each config file must contain the following parameters: train_file, test_file, num_classes, batch_size, epochs, learning_rate.
        """
        import toml

        if self.mlflow_enabled and evaluate:
            mlflow.start_run(run_name="evaluation_run")

        if self.mlflow_enabled and (not evaluate):
            mlflow.start_run(run_name="train_run")

        rootdir = path_to_config_directory
        for subdir, dirs, files in os.walk(rootdir):
            for file in files:
                path = os.path.join(subdir, file)
                if self.mlflow_enabled:
                    mlflow.log_artifact(path)

                with open(path, "r") as f:
                    if evaluate:
                        print("Fine-tuning: ", path)
                    else:
                        print("Training: ", path)
                    config = toml.load(f)
                    train_file = config["train_file"]
                    test_file = config["test_file"]
                    num_classes = config["num_classes"]
                    batch_size = config["batch_size"]

                    self.set_output_dimension(num_classes)
                    if num_classes != self.output_layer.get_dim():
                        raise ValueError("Output dimension is incorrect")

                    epochs = config["epochs"]
                    learning_rate = config["learning_rate"]

                    train_x, train_y = dataset.load_bolt_svm_dataset(
                        train_file, batch_size
                    )
                    test_x, test_y = dataset.load_bolt_svm_dataset(
                        test_file, batch_size
                    )

                    train_config = bolt.graph.TrainConfig.make(
                        learning_rate=learning_rate, epochs=1
                    )
                    predict_config = (
                        bolt.graph.PredictConfig.make()
                        .with_metrics(["categorical_accuracy"])
                        .silence()
                    )
                    for i in range(epochs):
                        self.model.train(train_x, train_y, train_config=train_config)
                        if verbose:
                            metrics = self.model.predict(
                                test_x, test_y, predict_config=predict_config
                            )
                            print(
                                "Epoch: ",
                                i + 1,
                                " Accuracy: ",
                                metrics[0]["categorical_accuracy"],
                                "\n",
                            )

                    metrics = self.model.predict(
                        test_x, test_y, predict_config=predict_config
                    )
                    print(
                        "Epoch: ",
                        i + 1,
                        " Accuracy: ",
                        metrics[0]["categorical_accuracy"],
                    )

                print("\n")

        if self.mlflow_enabled:
            save_loc = "./hidden_layer_parameters"
            self.hidden_layer.save_parameters(save_loc)
            mlflow.log_artifact(save_loc)
            mlflow.end_run()

    def evaluate(self, path_to_config_directory):
        self.eat_corpus(path_to_config_directory, evaluate=True)

    def download_hidden_parameters(self, link_to_parameter):
        local_param_path = mlflow.artifacts.download_artifacts(link_to_parameter)
        self.hidden_layer.load_parameters(local_param_path)
        print("Loaded parameters")
