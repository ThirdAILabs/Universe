import argparse
import json

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
        choices=list(runner_map.keys()),
        help="The runner to retrieve benchmark results for.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Regular expression indicating which configs to retrieve for the given runners.",  # Empty string returns all configs for the given runners.
    )
    parser.add_argument(
        "--config_type",
        type=str,
        default=None,
        help="If specified, will ensure that each config to be retrieve has a config_type field equal to this value.",
    )
    parser.add_argument(
        "--mlflow_uri",
        type=str,
        help="MLflow URI to read metrics from.",
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
        help="String of mlflow run name you want to compare to official runs",
    )

    # Do not set the following args when running this script manually, these are only for github actions
    parser.add_argument(
        "--official_slack_webhook",
        type=str,
        default="",
        help="Slack channel endpoint for official benchmarks",
    )
    parser.add_argument(
        "--branch_slack_webhook",
        type=str,
        default="",
        help="Slack channel endpoint for branch benchmarks",
    )
    parser.add_argument(
        "--branch_name",
        type=str,
        default="",
        help="Name of branch that benchmarks are being run on",
    )
    return parser.parse_args()


def process_mlflow_dataframe(mlflow_runs, num_runs, client, run_name=""):
    if run_name:
        # The regex finds a match for mlflow run name that starts with the given run_name arg
        # and has enough space at the end for the date (10 chars) because our mlflow callback
        # appends the date to run names.
        run_name_re = f"^{run_name}_.{{10}}"
        mlflow_runs = mlflow_runs[
            mlflow_runs["tags.mlflow.runName"].str.match(run_name_re)
        ]
    mlflow_runs = mlflow_runs[mlflow_runs["status"] == "FINISHED"]
    mlflow_runs = mlflow_runs[:num_runs]

    display_columns = ["start_time"]
    if "metrics.epoch_times" in mlflow_runs.columns:
        mlflow_runs["training_time"] = mlflow_runs.apply(
            lambda x: sum(
                [x.value for x in client.get_metric_history(x.run_id, "epoch_times")]
            ),
            axis=1,
        )
        display_columns += ["training_time"]

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

    print(mlflow_runs.apply(lambda x: x.start_time.date(), axis=1))

    # Convert the start time timestamp into a date to make it easier to read
    mlflow_runs["start_time"] = mlflow_runs.apply(lambda x: x.start_time.date(), axis=1)

    metric_columns = [col for col in mlflow_runs if col.startswith("metrics")]
    display_columns += metric_columns
    df = mlflow_runs[display_columns]
    df = df.rename(
        columns={
            col: ".".join(col.split(".")[1:])
            for col in df.columns
            if col.startswith("metrics")
        }
    )
    return df


def extract_mlflow_data(experiment_name, num_runs, run_name=""):
    mlflow.set_tracking_uri(args.mlflow_uri)
    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        return pd.DataFrame()
    exp_id = exp.experiment_id

    mlflow_runs = mlflow.search_runs(exp_id)
    df = process_mlflow_dataframe(mlflow_runs, num_runs, client, run_name)

    return df


if __name__ == "__main__":
    args = parse_arguments()

    # We set different arguments when we run on the main github branch, a different branch, or manually
    if args.branch_name == "main":
        # If running benchmarks on main, set preset arguments
        slack_webhook = args.official_slack_webhook
        args.num_official_runs = 5
        args.num_branch_runs = 0
    elif args.branch_name != "":
        # If running benchmarks on a branch other than main, set preset arguments
        slack_webhook = args.branch_slack_webhook
        args.run_name = args.branch_name
        args.num_official_runs = 3
        args.num_branch_runs = 2
    else:
        # If running benchmarks manually, we don't want to post to slack channels
        slack_webhook = ""

    for runner_name in args.runner:
        runner = runner_map[runner_name.lower()]

        configs = get_configs(
            runner=runner, config_regex=args.config, config_type=args.config_type
        )

        slack_payload_list = [""]
        slack_payload_idx = 0
        for config in configs:
            official_exp_name = f"{config.config_name}_benchmark"
            branch_exp_name = config.config_name

            official_runs = extract_mlflow_data(
                official_exp_name, args.num_official_runs
            )
            official_runs.insert(0, "exp_name", "official")
            runs = [official_runs]

            if args.num_branch_runs:
                branch_runs = extract_mlflow_data(
                    branch_exp_name, args.num_branch_runs, args.run_name
                )
                branch_runs.insert(0, "exp_name", args.run_name)
                runs.insert(0, branch_runs)

            df_md = pd.concat(runs).to_markdown(index=False)

            slack_payload_text = f"*{official_exp_name}* ```{df_md}``` \n"
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

        if slack_payload_list[0] != "":
            for payload in slack_payload_list:
                if slack_webhook:
                    requests.post(slack_webhook, json.dumps({"text": payload}))
                else:
                    print(payload)
