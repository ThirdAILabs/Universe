import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
import os
import toml
import requests
import json
from .runners.runner_map import runner_map
import re
import argparse

SLACK_WEBHOOK = (
    "https://hooks.slack.com/services/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"
)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Benchmark a dataset with Bolt")
    parser.add_argument(
        "--runner",
        type=str,
        required=True,
        choices=["udt", "bolt_fc", "dlrm"],
        help="Which runner to use to run the benchmark.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Regular expression indicating which configs to run for the given runner.",
    )
    parser.add_argument(
        "--mlflow_uri",
        type=str,
        help="MLflow URI to log metrics and artifacts.",
    )
    parser.add_argument(
        "--official_benchmark",
        action="store_true",
        help="Controls if the experiment is logged to the '_benchmark' experiment or the regular experiment. This should only be used for the github actions benchmark runner.",
    )
    return parser.parse_args()

def process_mlflow_dataframe(mlflow_runs, num_runs, client):
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


def extract_mlflow_data(experiment_name, num_runs=1, markdown=False):
    mlflow.set_tracking_uri(args.mlflow_uri)
    client = MlflowClient()
    exp_id = client.get_experiment_by_name(experiment_name).experiment_id

    mlflow_runs = mlflow.search_runs(exp_id)
    df = process_mlflow_dataframe(mlflow_runs, num_runs, client)

    if markdown:
        df = df.to_markdown(index=False)
    return df


if __name__ == "__main__":
    args = parse_arguments()

    runner = runner_map[args.runner.lower()]

    config_re = re.compile(args.config)
    configs = list(
        filter(
            lambda config: config_re.match(config.config_name),
            runner.config_type.__subclasses__(),
        )
    )
    if len(configs) == 0:
        raise ValueError(
            f"Could match regular expression '{args.config}' to any configs."
        )

    slack_payload_text = ""
    for config in configs:
        exp_name = f"{config.config_name}_benchmark" if args.official_benchmark else config.config_name

        df_md = extract_mlflow_data(exp_name, markdown=True)
        slack_payload_text += f"*{exp_name}* ```{df_md}``` \n"
        slack_payload = {"text": slack_payload_text}

    requests.post(SLACK_WEBHOOK, json.dumps(slack_payload))


