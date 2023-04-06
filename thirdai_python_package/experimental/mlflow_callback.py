import os
import platform
import socket
from typing import Any, Dict, Optional

from thirdai._thirdai import bolt


# TODO how can we define this under bolt.callbacks?
class MlflowCallback(bolt.callbacks.Callback):
    """An Mlflow callback is initialized for a single experiment run.
    Reusing an instance of MlflowCallback does not reset the run and instead
    logs params for the existing experiment.

    Args:
        tracking_uri: The uri that hosts the MLflow experiments.
        experiment_name: The name of the associated experiment (top-level
            header in Mlflow). Groups together runs with similar intent.
        run_name: Describes the run. Should include any details that don't
            fit in the experiment_args
        experiment_args: Dict[str, Any] Log parameters related to the
            configuration of the experiment. These are logged once at
            initialization. Examples include learning_rate, hidden_layer_dim, etc
        batch_interval: Optional[int] = None: If specified then the callback will
            log all batch metrics on this interval.
    """

    def __init__(
        self,
        tracking_uri: str,
        experiment_name: str,
        run_name: str,
        experiment_args: Dict[str, Any] = {},
        batch_interval: Optional[int] = None,
    ):
        super().__init__()
        import mlflow  # import inside class to not force another package dependency

        mlflow.set_tracking_uri(tracking_uri)
        experiment_id = mlflow.set_experiment(experiment_name)
        run_id = mlflow.start_run(run_name=run_name).info.run_id

        print(
            f"\nStarting Mlflow run at: \n{tracking_uri}/#/experiments/{experiment_id}/runs/{run_id}\n"
        )

        if experiment_args:
            for k, v in experiment_args.items():
                mlflow.log_param(k, v)

        self._log_machine_info()

        # TODO(david): how to log the commit we are on?
        # TODO(david): how to log the current file we ran this from?
        # TODO(david): what about credentials for this?
        # mlflow.log_artifact(__file__)

        self.batch_interval = batch_interval
        self.batch_cnt = 0

    def _log_machine_info(self):
        import mlflow  # import inside class to not force another package dependency
        import psutil

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

        mlflow.log_params(machine_info)

    def on_batch_end(self, model, train_state):
        if self.batch_interval == None:
            return

        self.batch_cnt += 1
        if self.batch_cnt % self.batch_interval == 0:
            import mlflow  # import inside class to not force another package dependency

            for name, values in train_state.get_all_train_metrics().items():
                mlflow.log_metric(self._clean(name), values[-1])
            mlflow.log_metric("learning_rate", train_state.learning_rate)

    def on_epoch_end(self, model, train_state):
        import mlflow  # import inside class to not force another package dependency

        for name, values in train_state.get_all_train_metrics().items():
            mlflow.log_metric(self._clean(name), values[-1])
        for name, values in train_state.get_all_validation_metrics().items():
            mlflow.log_metric("val_" + self._clean(name), values[-1])
        mlflow.log_metric("epoch_times", train_state.epoch_times[-1])
        mlflow.log_metric("learning_rate", train_state.learning_rate)

    def log_additional_metric(self, key, value, step=0):
        import mlflow  # import inside class to not force another package dependency

        mlflow.log_metric(key, value, step=step)

    def log_additional_param(self, key, value):
        import mlflow  # import inside class to not force another package dependency

        mlflow.log_param(key, value)

    def end_run(self):
        import mlflow  # import inside class to not force another package dependency

        mlflow.end_run()

    def _clean(self, metric_name):
        # mlflow doesn't like when metrics have "@" in them (e.g. "precision@k")
        return metric_name.replace("@", "-")
