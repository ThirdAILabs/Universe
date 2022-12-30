import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

# TODO: Consolidate this constant with existing configs
MLFLOW_URI = "http://ec2-184-73-150-35.compute-1.amazonaws.com"


def _process_mlflow_dataframe(mlflow_runs, num_runs):
    mlflow_runs = mlflow_runs[mlflow_runs["status"] == "FINISHED"]
    mlflow_runs = mlflow_runs[:num_runs]
    mlflow_runs["total_time"] = mlflow_runs.apply(
        lambda x: (x.end_time - x.start_time).total_seconds(), axis=1
    )

    metric_columns = [col for col in mlflow_runs if col.startswith("metrics")]
    display_columns = ["start_time", "total_time", "params.dataset"] + metric_columns
    df = mlflow_runs[display_columns]
    return df


def extract_mlflow_data(experiment_name, num_runs=5, markdown=False):
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient()
    exp_id = client.get_experiment_by_name(experiment_name).experiment_id

    mlflow_runs = mlflow.search_runs(exp_id)
    df = _process_mlflow_dataframe(mlflow_runs, num_runs)

    if markdown:
        df = df.to_markdown(index=False)
    return df
