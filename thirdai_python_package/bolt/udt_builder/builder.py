import pandas as pd
from openai import OpenAI


from .templates.model_templates import (
    supported_templates,
)
from .task_detector import (
    verify_dataframe,
    raise_exception_without_trace,
    auto_infer_model_builder,
    get_task_template_from_query,
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
            return detected_template.from_raw_types(
                target_column, dataframe, detected_template
            )

        if self.openai_client:
            detected_template = get_task_template_from_query(task, self.openai_client)

            if detected_template is None:
                return None

            return detected_template.from_raw_types(target_column, dataframe)

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
