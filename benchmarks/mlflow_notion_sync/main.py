import argparse
import time

from mlflow_notion import MLFlowReportGenerator, NotionFormatGenerator

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
            # create rows for each run
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
        # TODO support adding new columns
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


def main():
    parser = argparse.ArgumentParser(
        description="Synchronize Weekly Benchmarks between MlFlow and Notion"
    )

    parser.add_argument("--mlflow-uri", type=str, default="")

    mlflow_uri, notion_page_id, notion_token, benchmarks = parse_config(
        "benchmarks/mlflow_notion_sync/mlsync_config.yaml"
    )

    notion_report_format = get_report_format(
        "benchmarks/mlflow_notion_sync/report_format.yaml"
    )

    benchmark_experiment_report_format = parse_report_format(
        "benchmarks/mlflow_notion_sync/format.json"
    )
    mlflow_report_generator = MLFlowReportGenerator(
        uri=mlflow_uri,
        benchmarks=benchmarks,
        notion_report_format=notion_report_format,
        benchmark_experiment_report_format=benchmark_experiment_report_format,
    )

    # final_report = mlflow_report_generator.create_final_benchmark_report()

    # with open("benchmarks/mlflow_notion_sync/final_report.json", "w") as f:
    #     json.dump(final_report, f)

    notion_format_generator = NotionFormatGenerator(
        token_id=notion_token,
        page_id=notion_page_id,
        report_format=notion_report_format,
    )

    databases = notion_format_generator.retrieve_databases()
    # print(f"type of dbs = {type(databases)}")
    # print(f"databases = {databases}")

    past_report = notion_format_generator.format_in()

    while True:
        # pull from mlflow
        new_mlflow_report = mlflow_report_generator.create_final_benchmark_report()

        diff_report = compare_reports(
            previous_report=past_report, new_report=new_mlflow_report
        )

        print(f"DIFF REPORT = {diff_report}")
        print(f"PAST REPORT = {past_report}")
        print(f"NEW REPORT = {new_mlflow_report}")
        if diff_report:
            print("Adding new benchmark results to Notion ...")

            if diff_report["new"]:
                print("new experiments found. Adding benchmarks to Notion ...")
                synchronize_current_report_with_notion(
                    notion_format_generator=notion_format_generator,
                    report=new_mlflow_report,
                    diff_report=diff_report,
                    crud_command="create",
                )
            if diff_report["updated"]:
                print("updating existing benchmark experiments ...")
                synchronize_current_report_with_notion(
                    notion_format_generator=notion_format_generator,
                    report=new_mlflow_report,
                    diff_report=diff_report,
                    crud_command="update",
                )

        # time.sleep(2)
        break


if __name__ == "__main__":
    main()
