import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from utils import get_mlflow_uri


def _process_mlflow_dataframe(mlflow_runs, num_runs, client):
    mlflow_runs = mlflow_runs[mlflow_runs["status"] == "FINISHED"]
    mlflow_runs = mlflow_runs[:num_runs]

    mlflow_runs["training_time"] = mlflow_runs.apply(
        lambda x: sum(
            [x.value for x in client.get_metric_history(x.run_id, "epoch_times")]
        ),
        axis=1,
    )

    # Drop the epoch times column since it is no longer needed after calculating training time
    mlflow_runs.drop(columns=["metrics.epoch_times"], inplace=True)

    # Convert the start time timestamp into a date to make it easier to read
    mlflow_runs["start_time"] = mlflow_runs.apply(lambda x: x.start_time.date(), axis=1)

    metric_columns = [col for col in mlflow_runs if col.startswith("metrics")]
    display_columns = ["start_time", "training_time"] + metric_columns
    df = mlflow_runs[display_columns]
    df = df.rename(
        columns={
            col: col.split(".")[-1] for col in df.columns if col.startswith("metrics")
        }
    )
    return df


def extract_mlflow_data(experiment_name, num_runs=5, markdown=False):
    mlflow_uri = get_mlflow_uri()
    mlflow.set_tracking_uri(mlflow_uri)
    client = MlflowClient()
    exp_id = client.get_experiment_by_name(experiment_name).experiment_id

    mlflow_runs = mlflow.search_runs(exp_id)
    df = _process_mlflow_dataframe(mlflow_runs, num_runs, client)

    if markdown:
        df = df.to_markdown(index=False)
    return df
