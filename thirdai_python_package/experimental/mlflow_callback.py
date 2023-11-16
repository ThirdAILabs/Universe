import os
import platform
import socket
from typing import Any, Dict
import psutil

from thirdai._thirdai import bolt


class MlflowCallback(bolt.train.callbacks.Callback):
    def __init__(
        self,
        tracking_uri: str,
        experiment_name: str,
        run_name: str,
        experiment_args: Dict[str, Any] = {},
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

        _log_machine_info()

    def on_batch_end(self):
        import mlflow

        gb_used = psutil.Process().memory_info().rss / (1024**3)
        mlflow.log_metric("ram_used_gb", gb_used)

    def on_epoch_end(self):
        import mlflow  # import inside class to not force another package dependency

        for name, values in self.history.items():
            mlflow.log_metric(_clean(name), values[-1])

        mlflow.log_metric("learning_rate", self.train_state.learning_rate)

        gb_used = psutil.Process().memory_info().rss / (1024**3)
        mlflow.log_metric("ram_used_gb", gb_used)

    def log_additional_metric(self, key, value, step=0):
        import mlflow  # import inside class to not force another package dependency

        mlflow.log_metric(_clean(key), value, step=step)

        gb_used = psutil.Process().memory_info().rss / (1024**3)
        mlflow.log_metric("ram_used_gb", gb_used)

    def log_additional_param(self, key, value):
        import mlflow  # import inside class to not force another package dependency

        mlflow.log_param(_clean(key), value)

    def end_run(self):
        import mlflow  # import inside class to not force another package dependency

        mlflow.end_run()


def _log_machine_info():
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


def _clean(key):
    # mlflow doesn't like when metrics have "@", "(", or ")" in them (e.g. "precision@k")
    key = key.replace("(", "_")
    key = key.replace(")", "")
    key = key.replace("@", "_")
    return key
