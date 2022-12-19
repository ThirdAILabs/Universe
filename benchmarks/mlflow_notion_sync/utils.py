import json
import os
from typing import List, Tuple

import yaml


def parse_config(config_file_path: str) -> Tuple[str, str, str, List[str]]:
    """ """

    if not os.path.exists(config_file_path):
        raise ValueError(
            "Error Parsing MLflow-Notion Sync: Invalid Configuration File Path.\n"
        )
    with open(config_file_path) as mlflow_notion_config:
        config = yaml.safe_load(mlflow_notion_config)

    mlflow_uri = config["mlflow"]["uri"]
    notion_page_id = config["notion"]["page_id"]
    notion_token = config["notion"]["token"]
    benchmarks = config["benchmarks"]

    return mlflow_uri, notion_page_id, notion_token, benchmarks


def get_report_format(report_file_path: str) -> dict:
    if not os.path.exists(report_file_path):
        raise ValueError("File containing the notion report format must be provided.\n")
    with open(report_file_path) as report:
        report_format = yaml.safe_load(report)

    return report_format


def parse_report_format(json_report: str) -> dict:
    if not os.path.exists(json_report):
        raise ValueError(
            f"Error Reading the Format File: Invalid Path for File {json_report}"
        )

    with open(json_report) as report:
        notion_report_format = json.load(report)

    return notion_report_format


def compare_reports(previous_report, new_report):
    diff = {"new": {}, "deleted": {}, "updated": {}}
    if previous_report == new_report:
        return {}

    # if not previous_report:
    #     return new_report

    # compare old and new experiments
    for experiment_name in previous_report:
        # if the experiment is not in the new report, then it is deleted
        if experiment_name not in new_report:
            diff["deleted"][experiment_name] = {
                "deleted": list(previous_report[experiment_name]["runs"].keys())
            }
        # if the experiment is in the new report, then we will compare the runs
        else:
            new_experiment = new_report[experiment_name]
            prev_experiment = new_report[experiment_name]

            # check if they are not the same
            if new_experiment != prev_experiment:
                # compare the runs
                diff_run_report = {"new": [], "deleted": [], "updated": []}

                # compare the runs between the old an new report
                for run_id in prev_experiment["runs"]:
                    # if the run is not in the new report, then it is deleted
                    if run_id not in new_experiment["runs"]:
                        diff_run_report["deleted"].append(run_id)

                    # status of the run must have changed
                    elif (
                        new_experiment["runs"][run_id]
                        != prev_experiment["runs"][run_id]
                    ):
                        # updated rows
                        diff_run_report["updated"].append(run_id)

                # compare the runs between the new and old report
                for run_id in new_experiment["runs"]:
                    # if the run is not in the old, then it is added
                    if run_id not in prev_experiment["runs"]:
                        diff_run_report["new"].append(run_id)
                        # TODO check if the run has new metrics

                # add to the updated experiments
                diff["updated"][experiment_name] = diff_run_report

        # compare the new and old experiments
        for experiment_name in new_report:
            # if the experiment is not in the old report, then it is added
            if experiment_name not in previous_report:
                diff["new"][experiment_name] = experiment_name

    return diff
