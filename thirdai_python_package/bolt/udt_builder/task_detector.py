from __future__ import annotations
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
from openai import OpenAI
import pickle

from .templates import model_templates
from .templates.model_builder import ModelBuilder
from .column_inferencing import column_detector
import warnings

warnings.filterwarnings("ignore")


def raise_exception_without_trace(message):
    raise Exception(message) from None


def verify_dataframe(dataframe: pd.DataFrame, target_column_name: str, task: str):
    if target_column_name not in dataframe.columns:
        raise Exception("Specified target column not found in the dataframe")

    if len(dataframe) < 50 and task is None:
        raise Exception(
            f"Minimum required rows to infer the problem type is 50 but dataframe has number rows {len(dataframe)}"
        )

    if len(dataframe) == 0:
        raise Exception(f"Cannot detect a task for dataset with 0 rows.")


def auto_infer_model_builder(target_column_name: str, dataframe: pd.DataFrame):
    # approx representation of a column
    target_column = column_detector.detect_single_column_type(
        target_column_name, dataframe
    )

    input_columns = column_detector.get_input_columns(target_column_name, dataframe)

    if isinstance(target_column, column_detector.NumericalColumn):
        return ModelBuilder(
            dataframe,
            target_column,
            input_columns,
            model_templates.RegressionTemplate,
        )

    if isinstance(target_column, column_detector.CategoricalColumn):

        if target_column.number_tokens_per_row >= 4:

            token_column_candidates = (
                column_detector.get_token_candidates_for_token_classification(
                    target_column, input_columns
                )
            )

            if (
                len(token_column_candidates) == 1
                and target_column.unique_tokens_per_row * len(dataframe) < 250
            ):
                return ModelBuilder(
                    dataframe,
                    target_column,
                    input_columns,
                    model_templates.TokenClassificationTemplate,
                )

            source_column_candidates = (
                column_detector.get_source_column_for_query_reformulation(
                    target_column, input_columns
                )
            )

            if len(source_column_candidates) == 1 and target_column.token_type == "str":
                return ModelBuilder(
                    dataframe,
                    target_column,
                    input_columns,
                    model_templates.QueryReformulationTemplate,
                )

        return ModelBuilder(
            dataframe,
            target_column,
            input_columns,
            model_templates.TabularClassificationTemplate,
        )

    raise Exception(
        "Could not automatically infer task using the provided column name and the template. The following target types are supported for classification : Numerical, Categorical, and Text. Verify that the target column has one of the following types or explicitly specify the task. Check out https://github.com/ThirdAILabs/Demos/tree/main/universal_deep_transformer to learn more about how to initialize and train a UniversalDeepTransformer."
    )


class UDTBuilder:
    def __init__(
        self,
        dataset_path: str,
        target_column: str,
        task: str = None,
        openai_key: str = None,
    ):
        if openai_key is not None:
            print("Task detection using natural language enabled\n")

        self.openai_client = OpenAI(api_key=openai_key) if openai_key else None
        self.task_to_template_map = {
            template.task: template for template in model_templates.supported_templates
        }

        self.detect(dataset_path, target_column, task=task)

    def detect(self, dataset_path: str, target_column: str, task=None):
        df = pd.read_csv(dataset_path).dropna().astype(str)
        verify_dataframe(df, target_column, task)

        self.target_column = target_column
        self.dataframe = df

        template_names = "\n".join(
            f"• {name}" for name in self.task_to_template_map.keys()
        )

        detected_model_builder = self._get_model_builder(
            self.dataframe, self.target_column, task
        )
        if detected_model_builder is not None:

            print(
                f"Task detected: {detected_model_builder.task}\n"
                f"If this isn't the task you intended, you can:\n"
                f"1. Provide a more specific problem type, or \n"
                f"2. Choose from the following list of available tasks:\n"
                f"{template_names}\n"
                f"To detect a different task, call the detect function again with:\n"
                f"    bolt.UniversalDeepTransformer(\n"
                f"        dataset_path = {dataset_path},\n"
                f"        target_column = {target_column},\n"
                f"        task = 'your_selected_task' \n"
                f"    )\n"
            )
        else:
            print(
                f"Cannot detect the task. "
                f"To automatically detect a task, you can:\n"
                f"1. Provide a more specific problem type, or \n"
                f"2. Choose from the following list of available tasks:\n"
                f"{template_names}\n"
                f"For explicit task detection, call the detect function again with:\n"
                f"    bolt.UniversalDeepTransformer(\n"
                f"        dataset_path = {dataset_path},\n"
                f"        target_column = {target_column},\n"
                f"        task = 'your_selected_task' \n"
                f"    )",
            )

        self.model_builder = detected_model_builder

    def _get_model_builder(self, dataframe, target_column: str, task: str):

        if task == None:
            return auto_infer_model_builder(
                target_column_name=target_column, dataframe=dataframe
            )

        if task in self.task_to_template_map:
            detected_template = self.task_to_template_map[task]
            return ModelBuilder.from_raw_types(
                target_column, dataframe, detected_template
            )

        if self.openai_client:
            detected_template = model_templates.get_task_template_from_query(
                task, self.openai_client
            )

            if detected_template is None:
                return None, 0

            return ModelBuilder.from_raw_types(
                target_column, dataframe, detected_template
            )

        return None

    def build(self):
        if self.model_builder == None:
            raise Exception(
                "Cannot initialize a UniversalDeepTransformer with a NoneType model_builder. Ensure that the builder has detected a valid template before calling build on it."
            )
        try:
            self.model = self.model_builder.build()
        except Exception as ex:
            raise_exception_without_trace(ex.__str__())

        return self.model


def detect_and_build(
    dataset_path: str,
    target: str,
    task: str = None,
    openai_key: str = None,
    **kwargs,
):
    builder = UDTBuilder(
        dataset_path=dataset_path,
        target_column=target,
        task=task,
        openai_key=openai_key,
    )

    if builder.model_builder == None:
        raise Exception("Could not detect a valid task.")

    model = builder.build()
    return model
