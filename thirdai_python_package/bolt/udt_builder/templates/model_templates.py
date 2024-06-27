from __future__ import annotations
from abc import abstractclassmethod
import typing
from collections import defaultdict
import pandas as pd

from thirdai import bolt

from ..column_inferencing import column_detector


class UDTDataTemplate:
    task: str
    keywords: set
    description: str

    target_column_caster: function

    @staticmethod
    @abstractclassmethod
    def model_initialization_typehint(target_column_name) -> str:
        pass

    @staticmethod
    @abstractclassmethod
    def get_concrete_types(target_column, input_columns, dataframe):
        pass


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

    @staticmethod
    def model_initialization_typehint(target_column_name) -> str:
        typehint = f"""
        Use the below code snippet to explicitly instantiate a model for Tabular Classification.
        
        bolt.UniversalDeepTransformer(
            data_types = {{
                "{target_column_name}" : bolt.types.categorical(n_classes = 'number_unique_classes', type = 'str' or 'int', delimiter = 'specify value here' if multiclass classification else None),
                "text_column" : bolt.types.text(),
                "numerical_column" : bolt.types.numerical((min_value_in_column, max_value_in_column)),
                "date_column" : bolt.types.datetime(),
                
            }}
        )
        """
        return typehint

    @staticmethod
    def get_concrete_types(target_column, input_columns, dataframe):
        if target_column is None:
            raise Exception(
                f"Could not convert the specified target column into a valid categorical data type for TabularClassification."
            )

        data_types = {}
        data_types[target_column.column_name] = target_column

        for col in input_columns:
            data_types[col] = input_columns[col]

        return data_types


class RegressionTemplate(UDTDataTemplate):
    task = "regression"
    keywords = set(["regression"])
    description = "used to train a model for regression tasks and the output space is real numbers. can take arbitrary inputs like text numbers datetime etc."

    target_column_caster = column_detector.cast_to_numerical

    @staticmethod
    def model_initialization_typehint(target_column_name) -> str:
        typehint = f"""
        Use the below code snippet to explicitly instantiate a model for Tabular Classification.
        
        bolt.UniversalDeepTransformer(
            data_types = {{
                "{target_column_name}" : bolt.types.numerical((min_value_in_column, max_value_in_column)),
                "text_column" : bolt.types.text(),
                "numerical" : bolt.types.numerical((min_value_in_column, max_value_in_column)),
                "date_column" : bolt.types.datetime(),
                "categorical_column" : bolt.types.categorical(type = "str" or "int", delimiter = None if 1 category per row else "column delimiter")
            }}
        )
        """
        return typehint

    @staticmethod
    def get_concrete_types(target_column, input_columns, dataframe):
        if target_column is None:
            raise Exception(
                f"Could not convert the specified target column into a valid numerical data type for Regression."
            )

        data_types = {}
        data_types[target_column.column_name] = target_column

        for col in input_columns:
            data_types[col] = input_columns[col]

        return data_types


class TokenClassificationTemplate(UDTDataTemplate):
    task = "ner"
    keywords = set(
        ["named entity recognition", "ner", "pii", "pii redaction", "llm firewall"]
    )
    description = "used to train a token classification model. used to assign a label to each token in the sentence. can be used for ner pii etc. input is space seperated text tokens and output is space seperated labels."

    target_column_caster = column_detector.cast_to_categorical

    @staticmethod
    def model_initialization_typehint(target_column_name):
        typehint = f"""
        Use the below code snippet to explicitly instantiate a model for Token Classification.
        
        bolt.UniversalDeepTransformer(
            data_types = {{
                "{target_column_name}" : bolt.types.token_tags(default_tag = "default_tag", tags = List[named_tags]),
                "source_column_name" : bolt.types.text()
            }}
        )
        """
        return typehint

    @staticmethod
    def get_concrete_types(
        target,
        input_columns,
        dataframe,
    ):

        if target is None:
            raise Exception(
                f"Could not convert the specified target column into a valid categorical data type for Token Classification."
            )

        def get_target_tags(
            target: column_detector.CategoricalColumn, dataframe: pd.DataFrame
        ):
            tag_frequency_map = defaultdict(int)

            def add_key_to_dict(dc, key):
                if key.strip():
                    dc[key] += 1

            dataframe[target.column_name].apply(
                lambda row: [
                    add_key_to_dict(tag_frequency_map, key) for key in row.split(" ")
                ]
            )

            sorted_tags = sorted(
                tag_frequency_map.items(), key=lambda item: item[1], reverse=True
            )
            sorted_tag_keys = [tag for tag, freq in sorted_tags]

            return sorted_tag_keys

        detected_tags = get_target_tags(target, dataframe)

        data_types = {}
        data_types[target.column_name] = column_detector.TokenTags(
            column_name=target.column_name,
            default_tag=detected_tags[0],
            named_tags=detected_tags[1:],
        )

        if len(data_types[target.column_name].named_tags) > 250:
            raise Exception(
                f"Unexpected Number of Unique Tags Detected in the column {len(data_types[target.column_name].named_tags)} for Token Classification. Ensure that the column is the correct column for tags."
            )

        token_column_candidates = (
            column_detector.get_token_candidates_for_token_classification(
                target, input_columns
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

        data_types[token_column_candidates[0].column_name] = token_column_candidates[0]
        return data_types


class QueryReformulationTemplate(UDTDataTemplate):
    task = "query_reformulation"
    keywords = set(["reformulate", "query reformulation", "rephrase"])
    description = "used to train a model for query reformulation. pass in an input text and the output is also a text but reformulated. can be used for modifying grammatical errors in queries and other related tasks. can train in both supervised and unsupervised settings."

    target_column_caster = column_detector.cast_to_categorical

    @staticmethod
    def model_initialization_typehint(target_column_name):
        typehint = f"""
        Use the template below to explicitly instantiate a model for Query Reformulation. 
        
        bolt.UniversalDeepTransformer(
            data_types = {{
                "{target_column_name}" : bolt.types.text(),
                "source_column_name" : bolt.types.text()
            }}
        )
        """
        return typehint

    @staticmethod
    def get_concrete_types(
        target: column_detector.CategoricalColumn,
        input_columns: typing.Dict[str, column_detector.Column],
        dataframe: pd.DataFrame,
    ):

        if target is None:
            raise Exception(
                f"Could not convert the specified target column into a valid categorical data type for Token Classification."
            )

        token_column_candidates = (
            column_detector.get_source_column_for_query_reformulation(
                target, input_columns
            )
        )
        if len(token_column_candidates) == 0:
            raise Exception(
                "Could not find a valid source column for the target. Note that the number of tokens in each row in the source column should be comparable (tolerance : 33%) to the number of target in the corresponding target row."
            )

        if len(token_column_candidates) > 1:
            raise Exception(
                f"Found {len(token_column_candidates) } valid candidates for the token column in the dataset. The following columns are valid sources : {[column.column_name for column in token_column_candidates]}"
            )

        data_types = {}
        data_types[target.column_name] = column_detector.TextColumn(
            column_name=target.column_name
        )

        data_types[token_column_candidates[0].column_name] = token_column_candidates[0]
        return data_types


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
                
            }}
        )
        """
        return typehint

    @staticmethod
    def get_concrete_types(
        target: column_detector.CategoricalColumn,
        input_columns: typing.Dict[str, column_detector.Column],
        dataframe: pd.DataFrame,
    ):

        if target is None:
            raise Exception(
                f"Could not convert the specified target column into a valid sequence data type for RecurrentClassifier."
            )

        def get_maximum_tokens_in_row(
            column: column_detector.CategoricalColumn, dataframe: pd.DataFrame
        ):
            token_counts = dataframe[column.column_name].apply(
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
        data_types[target.column_name] = column_detector.SequenceType(
            column_name=target.column_name,
            delimiter=target.delimiter,
            estimated_n_classes=target.estimated_n_classes,
            max_length=get_maximum_tokens_in_row(target, dataframe),
        )

        for column_name, column in input_columns.items():
            if should_convert_to_sequence(column):
                data_types[column_name] = column_detector.SequenceType(
                    column_name=column_name,
                    delimiter=column.delimiter,
                )
            else:
                data_types[column_name] = input_columns[column_name]

        return data_types


class GraphClassificationTemplate(UDTDataTemplate):
    task = "graph_classification"
    keywords = set(["neighbours", "graph classification", "graph classifier"])
    description = "used for training models to perform classification tasks on graph-structured data. suitable for problems where the input data is represented as graphs, consisting of nodes and edges, and the goal is to predict a label for individual nodes. fraud review detection, social network entity classification, etc."

    target_column_caster = column_detector.cast_to_categorical

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
            }}
        )
        """

    @staticmethod
    def get_concrete_types(
        target: column_detector.CategoricalColumn,
        input_columns: typing.Dict[str, column_detector.Column],
        dataframe: pd.DataFrame,
    ):
        raise Exception(
            "Auto type inferencing for Graph Classifier Initialization is not yet supported. Refer to https://github.com/ThirdAILabs/Demos/blob/main/universal_deep_transformer/graph_neural_networks/GraphNodeClassification.ipynb on how to initialize a Graph Classifier."
        )


supported_templates = [
    TabularClassificationTemplate,
    RegressionTemplate,
    TokenClassificationTemplate,
    QueryReformulationTemplate,
    RecurrentClassifierTemplate,
    GraphClassificationTemplate,
]
