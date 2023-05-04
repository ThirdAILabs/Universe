import argparse
import json

import mlflow
import requests
from mlflow.tracking import MlflowClient
import pandas as pd

from .runners.runner_map import runner_map
from .utils import get_configs


def parse_arguments():
    parser = argparse.ArgumentParser(description="Benchmark a dataset with Bolt")
    parser.add_argument(
        "--runner",
        type=str,
        nargs="+",
        required=True,
        choices=["udt", "bolt_fc", "dlrm", "query_reformulation", "temporal"],
        help="The runner to retrieve benchmark results for.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Regular expression indicating which configs to retrieve for the given runners.",  # Empty string returns all configs for the given runners.
    )
    parser.add_argument(
        "--mlflow_uri",
        type=str,
        help="MLflow URI to read metrics from.",
    )
    parser.add_argument(
        "--official_benchmark",
        action="store_true",
        help="Controls if the experiments retrieved are '_benchmark' experiments or regular experiments. This should be used to retrieve experiments run by the github actions benchmark runner.",
    )
    parser.add_argument(
        "--num_official_runs",
        type=int,
        default=5,
        help="How many official runs to display in slack message",
    )
    parser.add_argument(
        "--num_branch_runs",
        type=int,
        default=0,
        help="How many branch runs to display in slack message",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="",
        help="Regular expression indicating which runs to retrieve for the branch benchmarks, only active when num_branch_runs > 0",
    )
    parser.add_argument(
        "--slack_webhook",
        type=str,
        default="",
        help="Slack channel endpoint for posting messages to. If this is empty, print to console",
    )
    return parser.parse_args()


def process_mlflow_dataframe(mlflow_runs, num_runs, client, run_name_regex=""):
    if run_name_regex:
        mlflow_runs = mlflow_runs[run_name_regex.match(mlflow_runs["tags.mlflow.runName"])]
    mlflow_runs = mlflow_runs[mlflow_runs["status"] == "FINISHED"]
    mlflow_runs = mlflow_runs[:num_runs]

    mlflow_runs["training_time"] = mlflow_runs.apply(
        lambda x: sum(
            [x.value for x in client.get_metric_history(x.run_id, "epoch_times")]
        ),
        axis=1,
    )

    # Drop the epoch times column since it is no longer needed after calculating training time
    mlflow_runs.drop(columns=["metrics.epoch_times"], inplace=True, errors="ignore")

    # Drop learning rate column since we don't need to display it as a recorded metric in Slack
    mlflow_runs.drop(columns=["metrics.learning_rate"], inplace=True, errors="ignore")

    # Drop test time because it is inconsistent between benchmarks due to some benchmarks using
    # model.evaluate vs regular callbacks. The average predict time metric can be used instead
    mlflow_runs.drop(columns=["metrics.test_time"], inplace=True, errors="ignore")
    mlflow_runs.drop(columns=["metrics.val_test_time"], inplace=True, errors="ignore")

    # Remove columns that contain only nan values, usually indicates deprecation of a metric
    mlflow_runs.dropna(axis=1, how="all", inplace=True)

    # Convert the start time timestamp into a date to make it easier to read
    mlflow_runs["start_time"] = mlflow_runs.apply(lambda x: x.start_time.date(), axis=1)

    metric_columns = [col for col in mlflow_runs if col.startswith("metrics")]
    display_columns = ["start_time", "training_time"] + metric_columns
    df = mlflow_runs[display_columns]
    df = df.rename(
        columns={
            col: ".".join(col.split(".")[1:])
            for col in df.columns
            if col.startswith("metrics")
        }
    )
    return df


def extract_mlflow_data(experiment_name, num_runs, run_name_regex=""):
    mlflow.set_tracking_uri(args.mlflow_uri)
    client = MlflowClient()
    exp_id = client.get_experiment_by_name(experiment_name).experiment_id

    mlflow_runs = mlflow.search_runs(exp_id)
    df = process_mlflow_dataframe(mlflow_runs, num_runs, client, run_name_regex)

    return df


if __name__ == "__main__":
    args = parse_arguments()

    print(args.slack_webhook)
