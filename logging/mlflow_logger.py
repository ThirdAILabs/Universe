import os
import platform
import socket
import time

import mlflow
import psutil
import toml

file_dir = os.path.dirname(os.path.abspath(__file__))
file_name = os.path.join(file_dir, "config.toml")
with open(file_name) as f:
    parsed_config = toml.load(f)
mlflow.set_tracking_uri(parsed_config["tracking"]["uri"])

# For load, see e.g. https://en.wikipedia.org/wiki/Load_(computing)
# For everythin else, https://stackoverflow.com/questions/3103178/how-to-get-the-system-info-with-python
machine_info = {
    "load_before_experiment": os.getloadavg()[2],
    "platform": platform.platform(),
    "platform_version": platform.version(),
    "platform_release": platform.release(),
    "architecture": platform.machine(),
    "processor": platform.processor(),
    "hostname": socket.gethostname(),
    "ram_gb": round(psutil.virtual_memory().total / (1024.0**3)),
    "num_cores": psutil.cpu_count(logical=True),
}


def _log_machine_info():
    for key, val in machine_info.items():
        mlflow.log_param(key, val)


class ExperimentLogger:
    """
    This class starts an mlflow run that logs metadata about a model run to
    our mlflow tracking server. It has fields for Bolt, but also can be used
    with Tensorflow for a direct comparison. Example usage:
      from mlflow_logger import ModelLogger
      with ModelLogger(
        dataset="amazon670k",
        learning_rate=0.01,
        num_hash_tables=10,
        hashes_per_table=5,
        sparsity=0.01,
        algorithm="bolt") as mlflow_logger:

        <Code to init model>

        mlflow_logger.log_start_training()

        for each epoch:
          <train model a single epoch>
          mlflow_logger.log_epoch(accuracy)

    To use with something like TensorFlow that does not have num_hash_tables
    , hashes_per_table, or sparsity, you can leave those fields out. Note
    mlflow only supports a single run at a time, so don't nest "with"
    statements containing this class or other mlflow logging classes
    (similarly don't have a nested call to log_magsearch).
    """

    def __init__(
        self,
        experiment_name,
        dataset,
        algorithm="bolt",
        experiment_args=None,
    ):
        self.experiment_name = experiment_name
        self.dataset = dataset
        self.algorithm = algorithm
        self.experiment_args = experiment_args
        self.epoch_times = []
        self.epoch_accuracies = []

    def __enter__(self):
        mlflow.set_experiment(self.experiment_name)
        mlflow.start_run(
            tags={"dataset": self.dataset, "algorithm": self.algorithm},
        )
        _log_machine_info()
        # mlflow.log_param("learning_rate", self.learning_rate)

        if self.experiment_args:
            for k, v in vars(self.experiment_args).items():
                mlflow.log_param(k, v)

        self.start_time = time.time()
        return self

    def log_start_training(self):
        """
        Call this method to log and store a start time for training, so
        the first epoch length can be recorded. Must be called before log_epoch.
        This also records the "initilization_time" as the time between
        the start of the "with" block and the start of this method.
        """
        self.epoch_times.append(time.time())
        mlflow.log_metric("accuracy", 0)
        mlflow.log_param("initilization_time", time.time() - self.start_time)

    def log_epoch(self, accuracy):
        """
        Logs epoch accuracy and epoch time for plotting accuracy vs. time
        graphs and getting average epoch time. Should be called as soon as
        an epoch finishes to get accurate time results, and log_start_training
        must have been called first.
        """
        if len(self.epoch_times) == 0:
            raise RuntimeError("Must call log_start_training before calling log_epoch.")

        self.epoch_times.append(time.time())
        self.epoch_accuracies.append(accuracy)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("epoch_length", self.epoch_times[-1] - self.epoch_times[-2])

    def log_final_accuracy(self, final_accuracy):
        """
        Call this method to log a final accuracy, usually by running the model
        on a larger final test set after training is complete.
        """
        mlflow.log_param("final_accuracy", final_accuracy)

    def __exit__(self, type, value, traceback):
        if len(self.epoch_accuracies) > 0 and len(self.epoch_times) > 1:
            mlflow.log_metric("final_accuracy", self.epoch_accuracies[-1])
            mlflow.log_param(
                "average_epoch_length",
                sum(self.epoch_times) / (len(self.epoch_times) - 1),
            )


# TODO: Change function signature
# def log_run(experiment, dataset, algorithm, *args, **kwargs)


def log_imagesearch_run(
    reservoir_size,
    hashes_per_table,
    num_tables,
    indexing_time,
    querying_time,
    num_queries,
    recall,
    dataset,
):
    """Starts and finishes an mlflow run for magsearch, logging all
    necessary information."""

    mlflow.set_experiment("Image Search")
    with mlflow.start_run(tags={"dataset": dataset, "algorithm": "magsearch"}):
        _log_machine_info()
        mlflow.log_param("reservoir_size", reservoir_size)
        mlflow.log_param("hashes_per_table", hashes_per_table)
        mlflow.log_param("num_tables", num_tables)
        mlflow.log_param("indexing_time", indexing_time)
        mlflow.log_param("querying_time", querying_time)
        mlflow.log_param("num_queries", num_queries)
        mlflow.log_param("queries_per_second", num_queries / querying_time)
        mlflow.log_param("recall", recall)
