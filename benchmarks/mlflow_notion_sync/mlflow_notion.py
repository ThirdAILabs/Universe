import json
import logging
import sys
from typing import List, Tuple
from urllib.parse import urlparse

import notion_client
import requests
import yaml

from utils import typify


class MLFlowExperimentsReport:
    """Generates a report for the benchmark experiments from MLFlow.

    Experiments reports are constructed as python dictionaries, and
    each report consists of the details of each run with the reported
    metrics.

    Args:
        - uri: MLFlow base URI
        - benchmarks: A list of benchmark experiments.
        - base_report_format: Custom report format for the experiments.
            This serves as a template report that can be modified
            depending on the parameters that a specific experiment
            has in a single experiment run.

    """

    MLFLOW_LIST_ENDPOINT = "/api/2.0/preview/mlflow/experiments/list"
    MLFLOW_SEARCH_ENDPOINT = "/api/2.0/mlflow/runs/search"
    MLFLOW_GET_HISTORY_ENDPOINT = "/api/2.0/mlflow/metrics/get-history"

    def __init__(
        self,
        uri,
        benchmarks: List[str],
        base_report_format: dict,
    ) -> None:
        self.base_uri = uri
        self.report_format = base_report_format

        self.benchmarks = benchmarks
        self.experiment_reports = {}
        self.__extract_benchmark_metadata()

    def __extract_benchmark_metadata(self) -> None:
        """
        Retrieves the metadata for all runs in each benchmark
        experiment specified.
        Using the `get-by-name` API endpoint would probably be
        a bit faster, but not significantly faster than using
        the `list` endpoint.
        """
        experiments_uri = self.base_uri + self.MLFLOW_LIST_ENDPOINT
        experiments_list = requests.get(experiments_uri).json()

        self.experiment_reports["experiments"] = []
        for experiment in experiments_list["experiments"]:
            if experiment["name"] in self.benchmarks:
                self.experiment_reports["experiments"].append(experiment)

    def __get_experiment_runs(
        self,
        experiment_id: str,
        max_results=100,
    ) -> dict:
        """
        Retrieves `max_results` runs for the experiment with the
        given ID.
        Args:
            - experiment_id: ID of the benchmark experiment
            - max_results: Maximum number of runs desired.

        For more parameters to use with the search endpoint, checkout
        https://www.mlflow.org/docs/latest/rest-api.html#search-experiments
        """

        endpoint = self.base_uri + self.MLFLOW_SEARCH_ENDPOINT
        runs = requests.post(
            endpoint,
            json={
                "experiment_ids": [experiment_id],
                "max_results": max_results,
                "page_token": None,
            },
        ).json()

        return runs

    def __get_benchmark_run_metrics(self, run_id: str, metric_key: str) -> dict:
        """
        Gets a list of all values for the specified metric for a given run.
        """
        endpoint_uri = self.base_uri + self.MLFLOW_GET_HISTORY_ENDPOINT
        return requests.get(
            endpoint_uri,
            json={"run_id": run_id, "metric_key": metric_key},
        ).json()

    def __generate_benchmark_experiment(self):
        report = {}
        experiment_report_format = self.report_format["experiment"]

        experiment_report_format_keys = experiment_report_format["values"].keys()

        for benchmark_experiment in self.experiment_reports["experiments"]:
            key = experiment_report_format["key"]

            benchmark_name = benchmark_experiment[key]
            # create a sub-report for this benchmark experiment
            report[benchmark_name] = {}

            for (key, value) in benchmark_experiment.items():
                # add the key to the report if it is in the format
                if key in experiment_report_format_keys:
                    alias = experiment_report_format["values"][key]["alias"]
                    report[benchmark_name][alias] = value

                else:
                    # If the key is not in the experiment report format, add it too.
                    report[benchmark_name][key] = value

        return report

    def __generate_benchmark_run_metrics(self, report_metric):
        if not report_metric:
            return {}

        metric_info = {}
        value, timestamp, step = [], [], []
        for metric in report_metric["metrics"]:
            value.append(metric["value"])
            timestamp.append(metric["timestamp"])
            step.append(metric["step"])

        metric_info["key"] = report_metric["metrics"][0]["key"]
        metric_info["value"] = (value,)
        metric_info["timestamp"] = timestamp
        metric_info["step"] = step

        return metric_info

    def create_final_benchmark_report(self) -> dict:
        """
        Generates the final report for each benchmark experiment.
        """

        runs = {
            experiment["experiment_id"]: self.__get_experiment_runs(
                experiment["experiment_id"]
            )
            for experiment in self.experiment_reports["experiments"]
        }

        report = self.__generate_benchmark_experiment()

        # loop through the experiments
        for _, benchmark_experiment in report.items():
            experiment_id = benchmark_experiment["experiment_id"]
            # create runs for each experiment
            benchmark_experiment["runs"] = {}
            reports_run = self.__generate_single_benchmark_run_report(
                runs=runs[experiment_id]
            )

            # add the runs to the experiment
            benchmark_experiment["runs"] = reports_run

        # step 4: generate the report
        report = {k: v for k, v in report.items() if v["runs"]}
        return report

    def __generate_single_benchmark_run_report(self, runs):
        report = {}

        elements = self.report_format["run"]["values"]

        # iterate through the run report
        for index, benchmark_run in enumerate(runs["runs"]):
            # step 1: create a unique id for the run
            run_key = self.report_format["run"]["key"]

            run_id = benchmark_run["info"][run_key]

            # create an entry in the table
            report[run_id] = {}

            # step 2: add elements to the report
            for key, value in benchmark_run["info"].items():
                if key in elements:
                    alias = elements[key]["alias"]
                    val_type = elements[key]["type"]
                    report[run_id][alias] = {
                        **elements[key],
                        "key": key,
                        "value": typify(value, val_type),
                    }

                else:
                    # add if unmatched
                    report[run_id][key] = {
                        "key": key,
                        "value": value,
                        "type": str(type(value)),
                    }

            # step 3: add other data to the report
            for element_type in ["metrics", "params", "tags"]:

                if "Name" not in report[run_id]:
                    report[run_id]["Name"] = {
                        "alias": "Name",
                        "type": "title",
                        "tag": "info",
                        "key": "id",
                        "description": "The unique ID of the run",
                        "value": "NULL",
                        "data": None,
                    }
                if "uid" not in report[run_id]:
                    report[run_id]["uid"] = {
                        "alias": "id",
                        "type": "string",
                        "tag": "info",
                        "key": "id",
                        "description": "Unique ID for the run",
                        "value": str(run_id),
                        "data": None,
                    }
                # make sure the element type is in the benchmark run
                if element_type in benchmark_run["data"]:
                    for metric in benchmark_run["data"][element_type]:
                        key, value = metric["key"], metric["value"]

                        # for each metric, detailed metrics may be available, so add them
                        metric_data = None
                        if element_type == "metrics":
                            # add detailed metrics
                            metric_data = self.__get_benchmark_run_metrics(
                                run_id=run_id, metric_key=metric["key"]
                            )
                            # post process the metric data
                            metric_data = self.__generate_benchmark_run_metrics(
                                metric_data
                            )

                        # check if the metric is part of elements
                        if key in elements:
                            alias = elements[key]["alias"]
                            val_type = elements[key]["type"]
                            report[run_id][alias] = {
                                **elements[key],
                                "key": key,
                                "value": typify(value, val_type),
                                "data": metric_data,
                            }
                        else:
                            report[run_id][key] = {
                                "key": key,
                                "value": value,
                                "type": str(type(value)),
                                "data": metric_data,
                            }

        return report


class NotionFormatGenerator:
    NOTION_PROPERTIES = {
        "title": {"title": {}},
        "int": {"number": {"format": "number"}},
        "float": {"number": {"format": "number"}},
        "number": {"number": {"format": "number"}},
        "rich_text": {"rich_text": {}},
        "select": {"select": {"options": []}},
    }

    def __init__(self, token_id, page_id) -> None:
        self.notion_client = notion_client.Client(auth=token_id, log_level=logging.INFO)
        self.__page_id = page_id
        self.__notion_state = {}

    @property
    def page_id(self):
        return self.__page_id

    @page_id.setter
    def page_id(self, page_id):
        self.__page_id = page_id

    @property
    def notion_state(self):
        return self.__notion_state

    @notion_state.setter
    def notion_state(self, state):
        self.__notion_state = state

    def __query_database(self, db_id):
        try:
            page_object = self.notion_client.pages.retrieve(self.__page_id)
        except Exception as exception:
            raise ValueError(f"Invalid Notion Page Id: {self.__page_id}")

        return page_object

    @staticmethod
    def convert_type_to_notion_style(value_type):
        number_types = {"<class 'float'>", "float", "int", "<class 'int'>", "integer"}
        rich_text_types = {"<class 'str'>", "str", "string", "timestamp"}

        if value_type in number_types:
            return "number"
        elif value_type in rich_text_types:
            return "rich_text"
        else:
            return value_type

    def _get_notion_property(self, property_type, metadata):
        property_type = self.convert_type_to_notion_style(property_type)

        if property_type in self.NOTION_PROPERTIES:
            notion_property = self.NOTION_PROPERTIES[property_type]
        else:
            sys.exit(f"unknown type of property: {property_type}")

        return notion_property

    def create_notion_property(self, property_type, metadata):
        property_type = self.convert_type_to_notion_style(property_type)

        property = {}
        if property_type == "title":
            property["title"] = [{"text": {"content": metadata}}]
        elif property_type == "select":
            property["select"] = {"name": metadata}
        elif property_type in ["number", "float", "int"]:
            property["number"] = metadata
        elif property_type == "rich_text":
            property["rich_text"] = [{"text": {"content": metadata}}]
        else:
            sys.exit(f"unknown type of property: {property_type}")

        return property

    def read_notion_property(self, property_object):
        if property_object["type"] == "rich_text":
            return property_object["rich_text"][0]["text"]["content"]
        elif property_object["type"] == "number":
            return property_object["number"]
        elif property_object["type"] == "title":
            return property_object["title"][0]["text"]["content"]
        elif property_object["type"] == "select":
            return property_object["select"]["name"]
        else:
            sys.exit(f"unknown property type: {property_object['type']}")

    def generate_notion_report(self):
        databases = self.retrieve_databases()

        state, report = {}, {}

        for database in databases["results"]:
            # make sure the database is in the root page
            if database["parent"]["page_id"] != self.__page_id:
                continue
            # If database is not empty
            if len(database["title"]) != 0:
                # get name and ID of the database
                db_name = database["title"][0]["text"]["content"]
                db_id = database["id"]

                # create experiment report
                experiment_report = {"name": db_name, "id": db_id, "runs": {}}
                # update notion state
                state[db_name] = {"database_id": db_id, "pages": {}}

                # get all pages (runs) in the database. This is within a try-except
                # block because notion_client may return a database_id even after
                # it has been deleted due to eventual consistency.
                # This block ensures that we only access databases (i.e., experiment)
                # that are still active.
                try:
                    pages = self.notion_client.databases.query(database_id=db_id)
                except Exception as e:
                    continue
                pages = pages["results"] if pages else []

                for page in pages:
                    page_id = page["id"]

                    page_uid = self.read_notion_property(page["properties"]["uid"])
                    page_properties = {}
                    for page_property_name, page_property in page["properties"].items():
                        # if page_property_name == "Name" and len(
                        #     page_property["title"] == 0
                        # ):
                        #     page_properties[page_property_name] = "NULL"
                        if (
                            page_property["type"] == "rich_text"
                            and len(page_property["rich_text"]) == 0
                        ):
                            continue
                        page_properties[page_property_name] = self.read_notion_property(
                            page_property
                        )

                    # update state
                    state[db_name]["pages"][page_uid] = {"page_id": page_id}
                    experiment_report["runs"][page_uid] = page_properties

                report[db_name] = experiment_report

        self.__notion_state = state
        return report

    def retrieve_databases(self) -> None:
        """
        For more info, check out the notion search API endpoint:
        https://developers.notion.com/reference/post-search

        The output of this API call is a dictionary with the following
        format:
        databases = {
            "object": "list",
            "results": [],
            "next_cursor": None,
            "has_more": False,
            "type": "page_or_database",
            "page_or_database": {}
        }

        """
        # fetch databases
        databases = self.notion_client.search(
            filter={"value": "database", "property": "object"}
        )
        return databases

    def create_database(self, name, properties, parent_page_id):
        """Create a new database in notion.

        Args:
            name (str): Name of the database
            properties (dict): Properties of the database
            parent_page_id (str): ID of the parent page.

        Note: The create database endpoint requires an integration
            to have insert content capabilities. Attempting to call
            the API without such capabilities will return an HTTP
            response with a 403 status code. For more possible return
            status codes, check out
            https://developers.notion.com/reference/create-a-database
        """
        parent = {"type": "page_id", "page_id": parent_page_id}
        title = [{"type": "text", "text": {"content": name}}]
        response = self.notion_client.databases.create(
            parent=parent, title=title, properties=properties
        )
        return response.get("id")

    def read_database(self, database_id):
        return self.notion_client.databases.query(database_id=database_id)

    def update_database(self, database_id, properties):
        response = self.notion_client.databases.update(
            database_id=database_id, properties=properties
        )
        return response["properties"]

    def convert_report_to_notion_table(self, report):
        notion_report = {}

        # each experiment becomes a database
        for experiment_name, experiment in report.items():

            experiment_report = {}

            # create the properties of the database
            experiment_property = {}
            # go through all the runs to create a superset of the properties
            for run_id, run in experiment["runs"].items():
                for key, value in run.items():
                    # add only newly encountered properites
                    if key not in experiment_property:
                        # check the type of the element

                        val_type = "title" if (key == "Name") else value["type"]
                        # Metadata needed for some types of properties
                        if val_type == "select":
                            metadata = value["options"]
                        else:
                            metadata = value["value"]

                        experiment_property[key] = self._get_notion_property(
                            val_type, metadata
                        )

            # Add the properties to the experiment report
            experiment_report["properties"] = experiment_property

            # create database rows
            run_properties = {}
            for run_uid, run in experiment["runs"].items():
                run_property = {}
                for key, value in run.items():
                    # update value type if needed
                    val_type = "title" if (key == "Name") else value["type"]
                    run_property[key] = self.create_notion_property(
                        val_type, value["value"]
                    )

                run_properties[run_uid] = run_property

            # add the runs to the experiment report
            experiment_report["rows"] = run_properties

            # add to the report
            notion_report[experiment_name] = experiment_report

        return notion_report

    def handle_crud_operation(
        self, database_id, properties={}, page_id=None, operation=None
    ):
        """
        Handles CRUD operations in a notion database. Databases in Notion consists
        of pages.
        Args:
            database_id (str): Database ID
            page_id (str): Page ID
            properties (dict): Page properties

        Note: For more on page properties, check out
        """
        parent = {"type": "database_id", "database_id": database_id}
        if operation == "create":
            response = self.notion_client.pages.create(
                parent=parent, properties=properties
            )
        elif operation == "delete":
            response = self.notion_client.pages.update(
                page_id, parent=parent, properties=properties, archived=True
            )
        elif operation == "update":
            response = self.notion_client.pages.update(
                page_id, parent=parent, properties=properties, archived=False
            )

        return response
