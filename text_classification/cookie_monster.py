from thirdai import bolt, dataset
from thirdai.dataset import DataPipeline, blocks
from thirdai.bolt import MlflowCallback

# Uncomment the following line when used on a machine with valid mlflow credentials
import mlflow
import os


class CookieMonster:
    def __init__(
        self,
        input_dimension,
        hidden_dimension=2048,
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
        self.mlm_loader = dataset.MLMDatasetLoader(
            self.input_dimension, masked_tokens_percentage=0.15
        )

        self.config_file_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_file_name = os.path.join(
            self.config_file_dir, "../benchmarks/config.toml"
        )
        with open(self.config_file_name) as f:
            parsed_config = toml.load(f)

        if self.mlflow_enabled:
            mlflow.set_tracking_uri(parsed_config["tracking"]["uri"])
            mlflow.set_experiment("Cookie Monster")

    # def setup_mlflow_tracking(self, run_name, output_dim):
    #     import toml

    #     with open(self.config_file_name) as file:
    #         parsed_config = toml.load(file)

    #     if self.mlflow_enabled:
    #         self.mlflow_callback = MlflowCallback(
    #             tracking_uri=parsed_config["tracking"]["uri"],
    #             experiment_name="Cookie Monster",
    #             run_name=run_name,
    #             dataset_name="wikipedia-20M",
    #             experiment_args={
    #                 "learning_rate": 0.0001,
    #                 "input_dim": self.input_dimension,
    #                 "hidden_dim": self.hidden_dim,
    #                 "output_dim": output_dim,
    #                 "hidden_sparsity": self.hidden_sparsity,
    #                 "output_layer_sparsity": 0.02,
    #             },
    #         )

    def get_mlflow_params(self, output_dim, learning_rate, epochs):
        return {
            "learning_rate": learning_rate,
            "epochs": epochs,
            "input_dim": self.input_dimension,
            "hidden_dim": self.hidden_dim,
            "output_dim": output_dim,
            "hidden_sparsity": self.hidden_sparsity,
            "output_layer_sparsity": 0.02,
        }

    def construct(self, output_dim):
        self.input_layer = bolt.graph.Input(dim=self.input_dimension)
        self.first_hidden_layer = bolt.graph.FullyConnected(
            dim=self.hidden_dim,
            sparsity=self.hidden_sparsity,
            activation="relu",
        )(self.input_layer)

        self.output_layer = bolt.graph.FullyConnected(
            dim=output_dim,
            activation="softmax",
            sparsity=0.02,
            sampling_config=bolt.DWTASamplingConfig(
                hashes_per_table=5, num_tables=128, reservoir_size=64
            ),
        )(self.first_hidden_layer)

        self.model = bolt.graph.Model(
            inputs=[self.input_layer], output=self.output_layer
        )

        self.model.compile(loss=bolt.CategoricalCrossEntropyLoss())

    def set_output_dimension(self, dimension):
        if self.output_layer.get_dim() == dimension:
            return
        first_hidden_save_loc = "./first_hidden_layer_parameters"
        # second_hidden_save_loc = "./second_hidden_layer_parameters"
        self.first_hidden_layer.save_parameters(first_hidden_save_loc)
        # self.second_hidden_layer.save_parameters(second_hidden_save_loc)

        self.construct(dimension)
        self.first_hidden_layer.load_parameters(first_hidden_save_loc)
        # self.second_hidden_layer.load_parameters(second_hidden_save_loc)
        os.remove(first_hidden_save_loc)
        # os.remove(second_hidden_save_loc)

    def load_data(self, task_type, file, batch_size, label_dim):
        if task_type == "mlm":
            # TODO: Check file format
            data, masked_indices, label = self.mlm_loader.load(file, batch_size)
        elif task_type == "classification":
            pipeline = DataPipeline(
                file,
                batch_size=batch_size,
                input_blocks=[blocks.TextPairGram(col=1, dim=self.input_dimension)],
                label_blocks=[blocks.NumericalId(col=0, n_classes=label_dim)],
                delimiter=",",
            )
            data, label = pipeline.load_in_memory()
        else:
            raise ValueError(
                'Invalid task_type. Supported task_types are "mlm" and "classification"'
            )
        return data, label

    def eat_corpus(
        self,
        path_to_config_directory,
        evaluate=False,
        verbose=False,
    ):
        """
        Given a directory containing only .txt config files, this function trains each dataset with the parameters specified in each config file.
        Each config file must contain the following parameters: train_file, test_file, num_classes, batch_size, epochs, learning_rate, task.
        """
        import toml

        if self.mlflow_enabled and evaluate:
            mlflow.start_run(run_name="fine-tuning")
            # self.setup_mlflow_tracking(run_name="fine-tuning")

        if self.mlflow_enabled and (not evaluate):
            mlflow.start_run(run_name="pre-training")
            # self.setup_mlflow_tracking(run_name="pre-training")

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
                        print("Pre-Training: ", path)
                    config = toml.load(f)
                    train_file = config["train_file"]
                    test_file = config["test_file"]
                    num_classes = config["num_classes"]
                    batch_size = config["batch_size"]
                    task = config["task"]

                    self.set_output_dimension(num_classes)
                    if num_classes != self.output_layer.get_dim():
                        raise ValueError("Output dimension is incorrect")

                    epochs = config["epochs"]
                    learning_rate = config["learning_rate"]

                    mlflow_params = self.get_mlflow_params(
                        num_classes, learning_rate, epochs
                    )
                    for param, val in mlflow_params.items():
                        mlflow.log_param(f"{param}", val)

                    train_config = (
                        bolt.graph.TrainConfig.make(
                            learning_rate=learning_rate, epochs=1
                        )
                        .with_rebuild_hash_tables(50000)
                        .with_reconstruct_hash_functions(10000)
                    )
                    predict_config = (
                        bolt.graph.PredictConfig.make()
                        .with_metrics(["categorical_accuracy"])
                        .silence()
                    )
                    train_x, train_y = self.load_data(
                        task, train_file, batch_size, num_classes
                    )
                    test_x, test_y = self.load_data(
                        task, test_file, batch_size, num_classes
                    )

                    for i in range(epochs):
                        train_output = self.model.train(
                            train_x, train_y, train_config=train_config
                        )
                        inference_output = self.model.predict(
                            test_x, test_y, predict_config=predict_config
                        )
                        print(
                            "Epoch: ",
                            i + 1,
                            " Accuracy: ",
                            inference_output[0]["categorical_accuracy"],
                            "\n",
                        )
                        train_metrics = {k: v[0] for k, v in train_output.items()}
                        mlflow.log_metrics(train_metrics)
                        mlflow.log_metrics(inference_output[0])

                print("\n")

        if self.mlflow_enabled:
            first_hidden_save_loc = "./first_hidden_layer_parameters"
            # second_hidden_save_loc = "./second_hidden_layer_parameters"
            self.first_hidden_layer.save_parameters(first_hidden_save_loc)
            # self.second_hidden_layer.save_parameters(second_hidden_save_loc)
            mlflow.log_artifact(first_hidden_save_loc)
            # mlflow.log_artifact(second_hidden_save_loc)
            mlflow.end_run()

    def evaluate(self, path_to_config_directory, evaluate=False):
        self.eat_corpus(path_to_config_directory, evaluate, verbose=True)

    def load_hidden_layer_parameters(
        self, first_hidden_layer_parameters, second_hidden_layer_parameters
    ):
        """
        Input:
            - hidden_layer_parameters: If MlFlow is enabled, this is a URL corresponding
                to where the parameters were saved. Otherwise, this is a path to a local
                file containing serialized parameters.

        """
        print("Loading Hidden Layer Parameters ...")
        if self.mlflow_enabled:
            local_param_path_hidden1 = mlflow.artifacts.download_artifacts(
                first_hidden_layer_parameters
            )
            local_param_path_hidden2 = mlflow.artifacts.download_artifacts(
                second_hidden_layer_parameters
            )

            print(
                "local parameter path first hidden layer = {}\n".format(
                    local_param_path_hidden1
                )
            )
            print(
                "local parameter path secon hidden layer = {}\n".format(
                    local_param_path_hidden2
                )
            )
            self.first_hidden_layer.load_parameters(
                "/home/blaise/first_hidden_layer_parameters"
            )
            self.second_hidden_layer.load_parameters(
                "/home/blaise/second_hidden_layer_parameters"
            )
        else:
            self.first_hidden_layer.load_parameters(first_hidden_layer_parameters)

        print("Loaded Hidden Layer Parameters")

    def download_hidden_parameters(self, link_to_parameter):
        local_param_path = mlflow.artifacts.download_artifacts(link_to_parameter)

        self.first_hidden_layer.load_parameters(local_param_path)
        print("Loaded parameters")
