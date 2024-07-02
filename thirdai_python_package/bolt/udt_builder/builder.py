import typing

import pandas as pd
from openai import OpenAI

from .task_detector import (
    auto_infer_task_template,
    get_task_template_from_query,
    raise_exception_without_trace,
    verify_dataframe,
)
from .templates.model_templates import UDTDataTemplate, supported_templates


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
        self.task_to_template_map: typing.Dict[str, UDTDataTemplate] = {
            template.task: template for template in supported_templates
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

        self.detected_template = self._get_template(
            self.dataframe, self.target_column, task
        )
        if self.detected_template is not None:
            print(
                f"Task detected: {self.detected_template.task}\n"
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

    def _get_template(self, dataframe, target_column: str, task: str):

        if task:
            if task in self.task_to_template_map:
                detected_template = self.task_to_template_map[task]
                return detected_template.from_raw_types(target_column, dataframe)

            if self.openai_client:
                detected_template = get_task_template_from_query(
                    task, self.openai_client
                )
                return detected_template.from_raw_types(target_column, dataframe)

            print(
                "Could not detect the task type with the provided type. Enabling auto-inference on the dataframe."
            )

        return auto_infer_task_template(
            target_column_name=target_column, dataframe=dataframe
        )

    def build(self):
        try:
            self.model = self.detected_template.build()
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

    if builder.detected_template == None:
        raise Exception("Could not detect a valid task.")

    model = builder.build()
    return model
