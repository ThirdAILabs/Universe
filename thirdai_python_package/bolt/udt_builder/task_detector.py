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


def auto_inference_model_builder(target_column_name: str, dataframe: pd.DataFrame):
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


class TemplateStore:
    def __init__(self, templates, openai_client: OpenAI):
        self.templates = templates
        self.openai_client = openai_client
        self.embedding_model = "text-embedding-3-large"
        self.embeddings = self._compute_embeddings()

    def _compute_embeddings(self):
        embeddings = {}
        for template in self.templates:
            text = f"task: {template.task}, description: {template.description}, keywords: {' '.join(template.keywords)}"
            embeddings[template] = self._get_embedding(text)
        return embeddings

    def _get_embedding(self, text):
        response = self.openai_client.embeddings.create(
            model=self.embedding_model, input=text
        )
        embedding = response.data[0].embedding
        return np.array(embedding)

    def find_closest_template(self, query):
        query_embedding = self._get_embedding(query)
        closest_template = None
        closest_distance = float("inf")
        all_distances = []

        for template, embedding in self.embeddings.items():
            distance = cosine(query_embedding, embedding)
            all_distances.append(distance)
            print(template.task, 1 - distance)
            if distance < closest_distance:
                closest_distance = distance
                closest_template = template

        mean_distance = np.median(all_distances)

        if (
            abs(mean_distance - closest_distance) < 0.05
            or (1 - closest_distance) < 0.35
        ):
            closest_template = None

        return closest_template, 1 - closest_distance

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path) -> TemplateStore:
        with open(path, "rb") as f:
            return pickle.load(f)


class UDTBuilder:
    def __init__(self, openai_key: str = None):
        self.template_store = (
            (
                TemplateStore(
                    model_templates.supported_templates,
                    openai_client=OpenAI(api_key=openai_key),
                )
            )
            if openai_key is not None
            else None
        )

        if openai_key is not None:
            print("Task detection using natural language enabled\n")

        self.task_to_template_map = {
            template.task: template for template in model_templates.supported_templates
        }

        self.detected_template = None
        self.model = None
        self.target_column = None
        self.dataframe = None

    def detect(self, dataset_path: str, target_column: str, task=None):
        df = pd.read_csv(dataset_path).dropna().astype(str)

        verify_dataframe(df, target_column, task)

        self.target_column = target_column
        self.dataframe = df

        template_names = "\n".join(
            f"• {name}" for name in self.task_to_template_map.keys()
        )

        # try:
        detected_model_builder, _ = self._detect_template(
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

    def _detect_template(self, dataframe, target_column: str, task: str):

        if task == None:
            return (
                auto_inference_model_builder(
                    target_column_name=target_column, dataframe=dataframe
                ),
                1,
            )

        if task in self.task_to_template_map:
            detected_template = self.task_to_template_map[task]
            return (
                ModelBuilder.from_raw_types(
                    target_column, dataframe, detected_template
                ),
                1,
            )

        if self.template_store is not None:
            detected_template, score = self.template_store.find_closest_template(task)

            if detected_template is None:
                return None, 0

            return (
                ModelBuilder.from_raw_types(
                    target_column, dataframe, detected_template
                ),
                score,
            )

        return None, 0

    def build(self):
        if self.model_builder == None:
            raise Exception(
                "Cannot initialize a UniversalDeepTransformer with a NoneType template. Ensure that the builder has detected a valid template before calling build on it."
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
    builder = UDTBuilder(openai_key=openai_key)
    builder.detect(dataset_path=dataset_path, target_column=target, task=task)

    if builder.model_builder == None:
        return

    model = builder.build()
    return model
