import argparse
import json
import os
import re

import mlflow
import pandas as pd
import requests
from mlflow.tracking import MlflowClient

from .runners.runner_map import runner_map
from .utils import get_configs


def parse_arguments():
    parser = argparse.ArgumentParser(description="Benchmark a dataset with Bolt")
    parser.add_argument(
        "--runner",
        type=str,
        nargs="+",
        required=True,
        choices=["udt", "bolt_fc", "dlrm"],
        help="The runner to retrieve benchmark results for.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Regular expression indicating which configs to retrieve for the given runner.",  # Empty string returns all configs for the given runner.
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
        "--num_runs",
        type=int,
        default=3,
        help="How many runs to display in slack message",
    )
    parser.add_argument(
        "--slack_webhook",
        type=str,
        default="",
        help="Slack channel endpoint to ping",
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

    # Drop learning rate column since we don't need to display it as a recorded metric in Slack

    mlflow_runs.drop(columns=["metrics.learning_rate"], inplace=True, errors='ignore')

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

    for runner_name in args.runner:

        runner = runner_map[runner_name.lower()]

        configs = get_configs(runner=runner, config_regex=args.config)

        slack_payload_list = [""]
        slack_payload_idx = 0
        for config in configs:
            exp_name = (
                f"{config.config_name}_benchmark"
                if args.official_benchmark
                else config.config_name
            )
            df_md = extract_mlflow_data(exp_name, num_runs=args.num_runs, markdown=True)

            slack_payload_text = f"*{exp_name}* ```{df_md}``` \n"
            line_length = len(slack_payload_text.split("\n")[0].split("```")[1])

            # We limit each message to under 4000 characters
            if (
                len(slack_payload_list[slack_payload_idx]) + len(slack_payload_text)
                >= 4000 - line_length
            ):
                slack_payload_list.append(slack_payload_text)
                slack_payload_idx += 1
            else:
                slack_payload_list[slack_payload_idx] += slack_payload_text

        for payload in slack_payload_list:
            requests.post(args.slack_webhook, json.dumps({"text": payload}))
