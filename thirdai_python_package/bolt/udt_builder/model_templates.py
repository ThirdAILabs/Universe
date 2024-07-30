from __future__ import annotations

import typing
from abc import abstractclassmethod

import pandas as pd
import thirdai._thirdai.bolt as bolt
from openai import OpenAI

from .column_inferencing import column_detector


class UDTDataTemplate:
    task: str
    keywords: set
    description: str

    target_column_caster: function

    demo_link: str = None

    @staticmethod
    @abstractclassmethod
    def model_initialization_typehint(target_column_name: str) -> str:
        pass

    @staticmethod
    @abstractclassmethod
    def get_concrete_types(
        target_column: column_detector.CategoricalColumn,
        input_columns: typing.Dict[str, column_detector.Column],
        dataframe: pd.DataFrame,
    ) -> typing.Tuple[column_detector.Column, typing.Dict[str, column_detector.Column]]:
        pass

    @property
    def target_column_name(self):
        return self.target_column.name

    @property
    def task(self):
        return self.task

    @property
    def bolt_data_types(self):
        return self._bolt_data_types

    @property
    def extreme_classification(self):
        if isinstance(self.target_column, column_detector.CategoricalColumn):
            if self.target_column.estimated_n_classes > 100_000:
                return True

        return False

    @classmethod
    def from_raw_types(cls, target_column_name, dataframe):
        raw_target_column = cls.target_column_caster(
            target_column_name, dataframe[target_column_name]
        )

        if raw_target_column is None:
            exception_string = f"Could not convert the specified target column {target_column_name} into a valid datatype for {cls.task}. Make sure that the target column name is correct and is a valid data type for the specified task. \n{cls.model_initialization_typehint(target_column_name)}."

            if cls.demo_link:
                exception_string += f"\nCheck out the demo notebook {cls.demo_link} for more information on how to initialize a UniversalDeepTransformer for the task."
            raise Exception(exception_string)

        raw_input_columns = column_detector.get_input_columns(
            target_column_name, dataframe
        )

        concrete_target_column, concrete_input_columns = cls.get_concrete_types(
            target_column=raw_target_column,
            input_columns=raw_input_columns,
            dataframe=dataframe,
        )

        return cls(dataframe, concrete_target_column, concrete_input_columns)

    def __init__(
        self,
        dataframe: pd.DataFrame,
        target_column: column_detector.Column,
        input_columns: typing.Dict[str, column_detector.Column],
    ):
        self.df = dataframe

        self.target_column = target_column
        self.input_columns = input_columns

        self._bolt_data_types = {}

        for column_name, column in self.input_columns.items():
            self._bolt_data_types[column_name] = column.to_bolt()

        self._bolt_data_types[self.target_column_name] = self.target_column.to_bolt(
            is_target_type=True
        )


class TabularClassificationTemplate(UDTDataTemplate):
    task = "tabular_classification"
    keywords = set(
        [
            "tabular classification",
            "sentiment classification",
            "product classification",
            "category classification",
        ]
    )
    description = "used to train a model for classification or prediction tasks with output being a label space of integers or strings such as sentiments, product ids, labels, document id, etc. can be used for sentiment classification, product classification, category classification. supports arbitrary inputs like datetime text sequences categorical data numbers."

    target_column_caster = column_detector.cast_to_categorical

    demo_link = "https://github.com/ThirdAILabs/Demos/blob/main/universal_deep_transformer/tabular_classification/FraudDetection.ipynb"

    @staticmethod
    def model_initialization_typehint(target_column_name: str) -> str:
        typehint = f"""
        Use the below code snippet to explicitly instantiate a model for Tabular Classification.
        
        bolt.UniversalDeepTransformer(
            data_types = {{
                "{target_column_name}" : bolt.types.categorical(n_classes = 'number_unique_classes', type = 'str' or 'int', delimiter = 'specify value here' if multiclass classification else None),
                "text_column" : bolt.types.text(),
                "numerical_column" : bolt.types.numerical((min_value_in_column, max_value_in_column)),
                "date_column" : bolt.types.datetime(),
                
            }},
            target = "{target_column_name}"
        )
        """
        return typehint

    @staticmethod
    def get_concrete_types(
        target_column: column_detector.CategoricalColumn,
        input_columns: typing.Dict[str, column_detector.Column],
        dataframe: pd.DataFrame,
    ):
        if target_column is None:
            raise Exception(
                f"Could not convert the specified target column into a valid categorical data type for TabularClassification."
            )

        data_types = {}

        for col in input_columns:
            data_types[col] = input_columns[col]

        return target_column, data_types


class RegressionTemplate(UDTDataTemplate):
    task = "regression"
    keywords = set(["regression"])
    description = "used to train a model for regression tasks and the output space is real numbers. can take arbitrary inputs like text numbers datetime etc."

    target_column_caster = column_detector.cast_to_numerical

    @staticmethod
    def model_initialization_typehint(target_column_name: str) -> str:
        typehint = f"""
        Use the below code snippet to explicitly instantiate a model for Tabular Classification.
        
        bolt.UniversalDeepTransformer(
            data_types = {{
                "{target_column_name}" : bolt.types.numerical((min_value_in_column, max_value_in_column)),
                "text_column" : bolt.types.text(),
                "numerical" : bolt.types.numerical((min_value_in_column, max_value_in_column)),
                "date_column" : bolt.types.datetime(),
                "categorical_column" : bolt.types.categorical(type = "str" or "int", delimiter = None if 1 category per row else "column delimiter")
            }},
            target = "{target_column_name}"
        )
        """
        return typehint

    @staticmethod
    def get_concrete_types(
        target_column: column_detector.CategoricalColumn,
        input_columns: typing.Dict[str, column_detector.Column],
        dataframe: pd.DataFrame,
    ):
        if target_column is None:
            raise Exception(
                f"Could not convert the specified target column into a valid numerical data type for Regression."
            )

        data_types = {}

        for col in input_columns:
            data_types[col] = input_columns[col]

        return target_column, data_types


class TokenClassificationTemplate(UDTDataTemplate):
    task = "ner"
    keywords = set(
        ["named entity recognition", "ner", "pii", "pii redaction", "llm firewall"]
    )
    description = "used to train a token classification model. used to assign a label to each token in the sentence. can be used for ner pii etc. input is space seperated text tokens and output is space seperated labels."

    target_column_caster = column_detector.cast_to_categorical

    demo_link = "https://github.com/ThirdAILabs/Demos/blob/main/universal_deep_transformer/named_entity_recognition/train_your_own_ner_model.ipynb"

    @staticmethod
    def model_initialization_typehint(target_column_name):
        typehint = f"""
        Use the below code snippet to explicitly instantiate a model for Token Classification.
        
        bolt.UniversalDeepTransformer(
            data_types = {{
                "{target_column_name}" : bolt.types.token_tags(default_tag = "default_tag", tags = List[named_tags]),
                "source_column_name" : bolt.types.text()
            }},
            target = "{target_column_name}"
        )
        """

        return typehint

    @staticmethod
    def get_concrete_target(
        target_column: column_detector.CategoricalColumn,
        dataframe: pd.DataFrame,
    ) -> column_detector.TokenTags:
        if target_column is None:
            raise Exception(
                f"Could not convert the specified target column into a valid categorical data type for Token Classification."
            )

        detected_tags = column_detector.get_frequency_sorted_unique_tokens(
            target_column, dataframe
        )

        concrete_target_column = column_detector.TokenTags(
            name=target_column.name,
            default_tag=detected_tags[
                0
            ],  # most occuring tag is the default tag for NER
            named_tags=detected_tags[1:],  # all other tags are named entities
        )

        if len(concrete_target_column.named_tags) > 250:
            raise Exception(
                f"Very High Number of Unique Tags for Token Classification. Number of Unique Tags Detected in the column {concrete_target_column.name} is {len(concrete_target_column.named_tags)}. Ensure that the column is the correct column for tags."
            )

        return concrete_target_column

    @staticmethod
    def get_concrete_types(
        target_column: column_detector.CategoricalColumn,
        input_columns: typing.Dict[str, column_detector.Column],
        dataframe: pd.DataFrame,
    ):

        data_types = {}

        concrete_target_column = TokenClassificationTemplate.get_concrete_target(
            target_column, dataframe
        )

        token_column_candidates = (
            column_detector.get_token_candidates_for_token_classification(
                target_column, input_columns
            )
        )

        if len(token_column_candidates) == 0:
            raise Exception(
                "Could not find a valid token column for the target. Note that the number of tokens in each row in the token column should be equal to the number of tags in the corresponding target row."
            )

        if len(token_column_candidates) > 1:
            raise Exception(
                f"Found {len(token_column_candidates) } valid candidates for the token column in the dataset. "
            )

        data_types[token_column_candidates[0].name] = token_column_candidates[0]
        return concrete_target_column, data_types


class QueryReformulationTemplate(UDTDataTemplate):
    task = "query_reformulation"
    keywords = set(["reformulate", "query reformulation", "rephrase"])
    description = "used to train a model for query reformulation. pass in an input text and the output is also a text but reformulated. can be used for modifying grammatical errors in queries and other related tasks. can train in both supervised and unsupervised settings."

    target_column_caster = column_detector.cast_to_categorical

    demo_link = "https://github.com/ThirdAILabs/Demos/blob/main/universal_deep_transformer/QueryReformulation.ipynb"

    @staticmethod
    def model_initialization_typehint(target_column_name):
        typehint = f"""
        Use the template below to explicitly instantiate a model for Query Reformulation. 
        
        bolt.UniversalDeepTransformer(
            data_types = {{
                "{target_column_name}" : bolt.types.text(),
                "source_column_name" : bolt.types.text()
            }},
            target = "{target_column_name}"
        )
        """
        return typehint

    @staticmethod
    def get_concrete_types(
        target_column: column_detector.CategoricalColumn,
        input_columns: typing.Dict[str, column_detector.Column],
        dataframe: pd.DataFrame,
    ):

        if target_column is None:
            raise Exception(
                f"Could not convert the specified target column into a valid categorical data type for Token Classification."
            )

        token_column_candidates = (
            column_detector.get_source_column_for_query_reformulation(
                target_column, input_columns
            )
        )
        if len(token_column_candidates) == 0:
            raise Exception(
                "Could not find a valid source column for the target. Note that the number of tokens in each row in the source column should be comparable (tolerance : 33%) to the number of target in the corresponding target row."
            )

        if len(token_column_candidates) > 1:
            raise Exception(
                f"Found {len(token_column_candidates) } valid candidates for the token column in the dataset. The following columns are valid sources : {[column.name for column in token_column_candidates]}"
            )

        data_types = {}

        concrete_target_column = column_detector.TextColumn(name=target_column.name)

        data_types[token_column_candidates[0].name] = token_column_candidates[0]
        return concrete_target_column, data_types


class RecurrentClassifierTemplate(UDTDataTemplate):
    task = "rnn"
    keywords = set(["recurrence", "rnn", "sequential", "sequence"])
    description = "used to train an rnn model. when you want to predict sequences. the output is a sequence of categories which can be both string and integer format. use cases : time series predictions, directions while navigating."
    requires_explict_instantiation = False

    target_column_caster = column_detector.cast_to_categorical

    @staticmethod
    def model_initialization_typehint(target_column_name):
        typehint = f"""
        Use the below code snippet to explicitly instantiate a model for Recurrent Classification.
        
        bolt.UniversalDeepTransformer(
            data_types = {{
                "{target_column_name}" : bolt.types.categorical(n_classes = 'number_unique_classes', type = 'str' or 'int', delimiter = 'delimiter', max_length = "maximum number of entities to predict in sequence"),
                "text_column" : bolt.types.text(),
                "numerical_column" : bolt.types.numerical((min_value_in_column, max_value_in_column)),
                "date_column" : bolt.types.datetime(),
                
            }},
            target = "{target_column_name}"
        )
        """
        return typehint

    @staticmethod
    def get_concrete_types(
        target_column: column_detector.CategoricalColumn,
        input_columns: typing.Dict[str, column_detector.Column],
        dataframe: pd.DataFrame,
    ):

        if target_column is None:
            raise Exception(
                f"Could not convert the specified target column into a valid sequence data type for RecurrentClassifier."
            )

        def get_maximum_tokens_in_row(
            column: column_detector.CategoricalColumn, dataframe: pd.DataFrame
        ):
            token_counts = dataframe[column.name].apply(
                lambda row: len(row.split(column.delimiter))
            )

            return max(token_counts)

        def should_convert_to_sequence(column: column_detector.Column):
            return (
                isinstance(column, column_detector.CategoricalColumn)
                and column.number_tokens_per_row >= 3
                and column.number_tokens_per_row <= 10
            )

        data_types = {}
        concrete_target_column = column_detector.SequenceType(
            name=target_column.name,
            delimiter=target_column.delimiter,
            estimated_n_classes=target_column.estimated_n_classes,
            max_length=get_maximum_tokens_in_row(target_column, dataframe),
        )

        for column_name, column in input_columns.items():
            if should_convert_to_sequence(column):
                data_types[column_name] = column_detector.SequenceType(
                    name=column_name,
                    delimiter=column.delimiter,
                )
            else:
                data_types[column_name] = input_columns[column_name]

        return concrete_target_column, data_types


class GraphClassificationTemplate(UDTDataTemplate):
    task = "graph_classification"
    keywords = set(["neighbours", "graph classification", "graph classifier"])
    description = "used for training models to perform classification tasks on graph-structured data. suitable for problems where the input data is represented as graphs, consisting of nodes and edges, and the goal is to predict a label for individual nodes. fraud review detection, social network entity classification, etc."

    target_column_caster = column_detector.cast_to_categorical

    demo_link = "https://github.com/ThirdAILabs/Demos/blob/main/universal_deep_transformer/graph_neural_networks/GraphNodeClassification.ipynb"

    @staticmethod
    def model_initialization_typehint(target_column_name):
        typehint = f"""
        Use the below code snippet to explicitly instantiate a model for Recurrent Classification.
        
        bolt.UniversalDeepTransformer(
            data_types = {{
                "{target_column_name}" : bolt.types.categorical(n_classes = 'number_unique_classes', type = 'str' or 'int', delimiter = 'delimiter', max_length = "maximum number of entities to predict in sequence"), # predicted class for the current node
                "node_id_column" : bolt.types.node_id(), # id of the current node
                "neighbour_column" : bolt.types.neighbors(), # space seperated ids
                
                "text_column" : bolt.types.text(),
                "numerical_column" : bolt.types.numerical((min_value_in_column, max_value_in_column)),
                "date_column" : bolt.types.datetime(),
            }},
            target = "{target_column_name}"
        )
        """

    @staticmethod
    def get_concrete_types(
        target_column: column_detector.CategoricalColumn,
        input_columns: typing.Dict[str, column_detector.Column],
        dataframe: pd.DataFrame,
    ):
        raise Exception(
            f"Auto type inferencing for Graph Classifier Initialization is not yet supported. Refer to {GraphClassificationTemplate.demo_link} on how to initialize a Graph Classifier."
        )


SUPPORTED_TEMPLATES = [
    TabularClassificationTemplate,
    RegressionTemplate,
    TokenClassificationTemplate,
    QueryReformulationTemplate,
    RecurrentClassifierTemplate,
    GraphClassificationTemplate,
]

TASK_TO_TEMPLATE_MAP: typing.Dict[str, UDTDataTemplate] = {
    template.task: template for template in SUPPORTED_TEMPLATES
}

SUPPORTED_TASK_TYPES = list(TASK_TO_TEMPLATE_MAP.keys())
