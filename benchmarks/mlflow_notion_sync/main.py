import argparse
import json
import time

from mlflow_notion import MLFlowExperimentsReport, NotionFormatGenerator

from utils import compare_reports, get_report_format, parse_config, parse_report_format


def synchronize_current_report_with_notion(
    notion_format_generator, report, diff_report, command
):
    notion_report = notion_format_generator.convert_report_to_notion_table(
        report=report
    )

    if command == "new":
        # create new tables for all experiments
        for experiment_name, experiment in notion_report.items():
            # check if the experiment is empty
            if not experiment["properties"]:
                continue
            database_id = notion_format_generator.create_database(
                name=experiment_name,
                properties=experiment["properties"],
                parent_page_id=notion_format_generator.page_id,
            )
            # add to notion state
            notion_format_generator.notion_state[experiment_name] = {
                "database_id": database_id,
                "pages": {},
            }
            # create rows for each run
            for run_uid, run in experiment["rows"].items():
                response = notion_format_generator.handle_crud_operation(
                    database_id=database_id, properties=run, operation="create"
                )
                page_id = response["id"]
                notion_format_generator[experiment_name]["pages"][run_uid] = {
                    "page_id": page_id
                }

    elif command == "create":
        assert diff_report is not None, "diff_report is required for the create command"
        # create tables for all experiments in the diff
        for experiment_name in diff_report["new"]:
            # check if the experiment is empty
            if not notion_report[experiment_name]["properties"]:
                continue
            # create new table
            database_id = notion_format_generator.create_database(
                experiment_name,
                notion_report[experiment_name]["properties"],
                notion_format_generator.page_id,
            )
            # add to notion state
            notion_format_generator.notion_state[experiment_name] = {
                "database_id": database_id,
                "pages": {},
            }
            # # create rows for each run
            for run_uid, run in notion_report[experiment_name]["rows"].items():
                response = notion_format_generator.handle_crud_operation(
                    database_id=database_id, properties=run, operation="create"
                )
                page_id = response["id"]
                notion_format_generator[experiment_name]["pages"][run_uid] = {
                    "page_id": page_id
                }

    elif command == "update":
        # New runs are added at the end of the table
        assert diff_report is not None

        # update existing tables for all experiments
        for experiment_name in diff_report["updated"]:
            # get existing database id
            database_id = notion_format_generator[experiment_name]["database_id"]
            # check if any new fields (columns) are added. If so, we need to update the
            # database first.
            current_database = notion_format_generator.read_database(
                database_id=database_id
            )
            current_properties = (
                current_database["results"][0]["properties"]
                if (current_database["results"])
                else {}
            )
            # check if there are new properties
            new_properties = {
                k: v
                for k, v in notion_report[experiment_name]["properties"].items()
                if k not in current_properties
            }
            if new_properties:
                # update the database
                notion_format_generator.update_database(
                    database_id=database_id, properties=new_properties
                )
            # add new rows
            for run_uid in diff_report["updated"][experiment_name]["new"]:
                run = notion_report[experiment_name]["rows"][run_uid]
                response = notion_format_generator.handle_crud_operations(
                    database_id=database_id, properties=run, operation="create"
                )
                page_id = response["id"]
                # add to notion state
                notion_format_generator.notion_state[experiment_name]["pages"][
                    run_uid
                ] = {"page_id": page_id}
            # update existing rows
            for run_uid in diff_report["updated"][experiment_name]["updated"]:
                run = notion_report[experiment_name]["rows"][run_uid]
                page_id = notion_format_generator.notion_state[experiment_name][
                    "pages"
                ][run_uid]["page_id"]

            # update notion
            notion_format_generator.handle_crud_operation(
                database_id=database_id,
                properties=run,
                page_id=page_id,
                operation="udpate",
            )


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Synchronize Weekly Benchmarks between MlFlow and Notion"
    )

    parser.add_argument("--mlflow_uri", type=str, help="The URI for MLFlow Experiments")
    parser.add_argument(
        "--auth_config",
        type=str,
        default="benchmarks/mlflow_notion_sync/config.yaml",
        help="Path to a config yaml file with credentials used to authenticate Notion and MLFlow API calls.",
    )

    parser.add_argument(
        "--report_format",
        type=str,
        default="benchmarks/mlflow_notion_sync/mlsync_config.yaml",
        help="Path to the base format used to synchronize runs from MLFlow and Notion.",
    )

    args = parser.parse_args()
    return args


def main():

    args = parse_arguments()
    mlflow_uri, notion_page_id, notion_token, benchmarks = parse_config(
        args.auth_config
    )

    report_format = get_report_format(args.report_format)

    mlflow_report_generator = MLFlowExperimentsReport(
        uri=mlflow_uri, benchmarks=benchmarks, base_report_format=report_format
    )

    notion_format_generator = NotionFormatGenerator(
        token_id=notion_token,
        page_id=notion_page_id,
    )

    previous_report = notion_format_generator.generate_notion_report()

    # pull from mlflow
    new_mlflow_report = mlflow_report_generator.create_final_benchmark_report()

    diff_report = compare_reports(
        previous_report=previous_report, new_report=new_mlflow_report
    )

    if diff_report:
        if len(diff_report["new"]) != 0:
            synchronize_current_report_with_notion(
                notion_format_generator=notion_format_generator,
                report=new_mlflow_report,
                diff_report=diff_report,
                command="create",
            )
        elif len(diff_report["updated"]) != 0:
            synchronize_current_report_with_notion(
                notion_format_generator=notion_format_generator,
                report=new_mlflow_report,
                diff_report=diff_report,
                command="update",
            )


if __name__ == "__main__":
    main()
