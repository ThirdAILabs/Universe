import os
from abc import abstractmethod

import pandas as pd
import pytest
from openai import OpenAI
from thirdai import bolt
from thirdai.bolt.udt_modifications import task_detector

from udt_builder_utils import *

OPENAI_KEY = os.getenv("OPENAI_API_KEY", None)


@pytest.fixture(scope="module")
def data_file():
    # Prepare the data
    dataframe = pd.DataFrame(
        {
            "float_col": get_numerical_column(100, 0, 10_000_000, create_floats=True),
            "int_col": get_numerical_column(100, 0, 10_000_000, create_floats=False),
            # for tabular classification
            "int_categorical_col": get_numerical_column(
                100, 0, 10000, create_floats=False
            ),
            "string_categorical_col": get_string_categorical_column(
                100, 1, delimiter=""
            ),
            "int_multi_categorical_col": get_int_categorical_column(
                100, 3, 0, 1000, ":"
            ),
            "string_multi_categorical_col": get_string_categorical_column(100, 3, " "),
            # for token classification
            "ner_source_col": get_string_categorical_column(100, 5, " "),
            "ner_target_col": get_string_categorical_column(
                100, 5, " ", select_tokens_from=["O", "A", "B"]
            ),
            # for query reformulation
            "query_reformulation_source": get_string_categorical_column(100, 11, " "),
            "query_reformulation_target": get_string_categorical_column(100, 10, " "),
        }
    )
    file_path = "temp.csv"
    dataframe.to_csv(file_path, index=False)

    yield file_path

    # cleanup
    import os

    os.remove(file_path)


class CommonTemplate:
    task: str
    query: str

    @abstractmethod
    def verify_template(self, template):
        pass

    def run_task_inference_and_training(self, data_file, target_col, task):
        detected_template = task_detector.detect_template(
            data_file, target_col, task=task, openai_key=OPENAI_KEY
        )
        self.verify_template(detected_template)

        model = bolt.UniversalDeepTransformer(
            target=target_col, data_types=detected_template.bolt_data_types
        )

        model.train("temp.csv", epochs=1, learning_rate=1e-3)

    def test_automatic_task_inference(self, data_file, target_col):
        self.run_task_inference_and_training(data_file, target_col, None)

    def test_explicit_task_inference(self, data_file, target_col):
        self.run_task_inference_and_training(data_file, target_col, self.task)

    def test_task_inference_using_natural_language(self, data_file, target_col):
        detected_template = task_detector.get_template_from_query(
            query=self.query,
            openai_client=OpenAI(api_key=OPENAI_KEY),
            target_column=target_col,
            dataframe=pd.read_csv(data_file).astype(str),
        )
        self.verify_template(detected_template)


@pytest.mark.parametrize("target_col", ["float_col", "int_col"])
class TestRegressionTemplate(CommonTemplate):
    task = task_detector.RegressionTemplate.task
    query = "i want to assign a score to these inputs"

    @classmethod
    def verify_template(cls, template):
        assert isinstance(template, task_detector.RegressionTemplate)
        assert template.extreme_classification == False
        assert isinstance(
            template.target_column, task_detector.column_detector.NumericalColumn
        )
        assert template.target_column.maximum < 10_000_000
        assert template.target_column.minimum >= 0


@pytest.mark.parametrize(
    "target_col",
    [
        "string_categorical_col",
        "int_categorical_col",
        "int_multi_categorical_col",
        "string_multi_categorical_col",
    ],
)
class TestTabularClassificationTemplate(CommonTemplate):
    task = task_detector.TabularClassificationTemplate.task
    query = "i have some product ids and want to map inputs to these products."

    @classmethod
    def verify_template(cls, template):

        assert isinstance(template, task_detector.TabularClassificationTemplate)
        assert template.extreme_classification == False
        assert isinstance(
            template.target_column, task_detector.column_detector.CategoricalColumn
        )


@pytest.mark.parametrize("target_col", ["ner_target_col"])
class TestTokenClassificationTemplate(CommonTemplate):
    task = task_detector.TokenClassificationTemplate.task
    query = "i have some tokens and the correspoding tags"

    @classmethod
    def verify_template(cls, template):
        assert isinstance(template, task_detector.TokenClassificationTemplate)
        assert isinstance(
            template.target_column, task_detector.column_detector.TokenTags
        )

        detected_tags = template.target_column.named_tags
        detected_tags.append(template.target_column.default_tag)

        assert set(detected_tags) == set(["O", "A", "B"])

        assert (
            len(template.input_columns) == 1
            and "ner_source_col" in template.input_columns.keys()
            and isinstance(
                template.input_columns["ner_source_col"],
                task_detector.column_detector.TextColumn,
            )
        )


@pytest.mark.parametrize("target_col", ["query_reformulation_target"])
class TestQueryReformulationTemplate(CommonTemplate):
    task = task_detector.QueryReformulationTemplate.task
    query = "i have some wrong sentences and i want to correct them"

    @classmethod
    def verify_template(cls, template):
        assert isinstance(template, task_detector.QueryReformulationTemplate)
        assert isinstance(
            template.target_column, task_detector.column_detector.TextColumn
        )

        assert (
            len(template.input_columns) == 1
            and "query_reformulation_source" in template.input_columns.keys()
            and isinstance(
                template.input_columns["query_reformulation_source"],
                task_detector.column_detector.TextColumn,
            )
        )


@pytest.mark.parametrize(
    "target_col", ["int_multi_categorical_col", "string_multi_categorical_col"]
)
class TestRecurrentClassificationTemplate(CommonTemplate):
    task = task_detector.RecurrentClassifierTemplate.task
    query = "i have some some inputs and i want to map them to a target sequence"

    @classmethod
    def verify_template(cls, template):

        assert isinstance(template, task_detector.RecurrentClassifierTemplate)
        assert template.extreme_classification == False
        assert isinstance(
            template.target_column, task_detector.column_detector.SequenceType
        )

        assert template.target_column.max_length == 3

    def test_automatic_task_inference(self, data_file, target_col):
        pass
