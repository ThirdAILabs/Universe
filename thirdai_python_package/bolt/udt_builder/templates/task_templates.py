import typing
from collections import defaultdict
import copy

import pandas as pd
import thirdai._thirdai.bolt as bolt

from ..column_inferencing import column_detector


def get_input_columns(target_column_name, dataframe: pd.DataFrame) -> typing.Dict:
    input_data_types = {}

    for col in dataframe.columns:
        if col == target_column_name:
            continue

        input_data_types[col] = column_detector.detect_single_column_type(
            column_name=col, dataframe=dataframe
        )

    return input_data_types


def get_token_candidates_for_token_classification(
    target: column_detector.CategoricalColumn,
    input_columns: typing.Dict[str, column_detector.Column],
) -> column_detector.TextColumn:

    if target.delimiter != " ":
        raise Exception("Expected the target column to be space seperated tags.")

    candidate_columns: typing.List[column_detector.Column] = []
    for column_name, column in input_columns.items():
        if isinstance(column, column_detector.CategoricalColumn):
            if (
                column.delimiter == " "
                and abs(column.number_tokens_per_row - target.number_tokens_per_row)
                < 0.001
            ):
                candidate_columns.append(
                    column_detector.TextColumn(column_name=column.column_name)
                )
    return candidate_columns


def get_source_column_for_query_reformulation(
    target: column_detector.CategoricalColumn,
    input_columns: typing.Dict[str, column_detector.Column],
) -> column_detector.TextColumn:

    if target.delimiter != " ":
        raise Exception("Expected the target column to be space seperated tags.")

    candidate_columns: typing.List[column_detector.CategoricalColumn] = []
    for column_name, column in input_columns.items():
        if isinstance(column, column_detector.CategoricalColumn):
            if column.delimiter == " ":
                ratio_source_to_target = (
                    column.number_tokens_per_row / target.number_tokens_per_row
                )
                if ratio_source_to_target > 1.5 or ratio_source_to_target < 0.66:
                    continue

                candidate_columns.append(
                    column_detector.TextColumn(column_name=column.column_name)
                )

    return candidate_columns


class ModelBuilderTemplate:

    @staticmethod
    def get_target_column(
        target_column_name, dataframe: pd.DataFrame, casting_function
    ):
        try:
            column = casting_function(target_column_name, dataframe[target_column_name])
            return column
        except:
            return None

    def build(self):
        extreme_classification = False

        target_column = self.target_column
        if isinstance(target_column, column_detector.CategoricalColumn):
            if target_column.estimated_n_classes > 100_000:
                extreme_classification = True

        self.model = bolt.UniversalDeepTransformer(
            data_types=self.bolt_data_types,
            target=self.target_column_name,
            extreme_classification=extreme_classification,
        )

        return self.model

    @property
    def target_column(self):
        return copy.copy(self.concrete_types[self.target_column_name])

    @property
    def input_columns(self):
        cols = {}
        for name, col in self.concrete_types.items():
            if name != self.target_column_name:
                cols[name] = col
        return cols


class TabularClassificationTemplate(ModelBuilderTemplate):
    name = "tabular_classification"
    keywords = set(
        [
            "tabular classification",
            "sentiment classification",
            "product classification",
            "category classification",
        ]
    )
    description = "used to train a model for classification or prediction tasks with output being a label space of integers or strings such as sentiments, product ids, labels, document id, etc. can be used for sentiment classification, product classification, category classification. supports arbitrary inputs like datetime text sequences categorical data numbers."

    requires_explicit_instantiaion = False

    def __init__(self, target_column_name, dataframe) -> None:
        self.target_column_name = target_column_name
        self.df = dataframe

        target_column = self.get_target_column(
            self.target_column_name,
            self.df,
            casting_function=column_detector.cast_to_categorical,
        )
        input_columns = get_input_columns(self.target_column_name, self.df)

        self.concrete_types = self.get_concrete_types(target_column, input_columns)

        self.bolt_data_types = {}

        for column_name, column in self.concrete_types.items():
            if column_name == target_column.column_name:
                self.bolt_data_types[column_name] = column.to_bolt(is_target_type=True)
            else:
                self.bolt_data_types[column_name] = column.to_bolt()

        if target_column.estimated_n_classes > 1_000_000:
            raise Exception(
                f"The dimension of the output label space is detected to be more than {1_000_000}. Verify that the target column is correctly specified."
            )

        if target_column.estimated_n_classes > 100_000:
            print(
                f"The dimension of the output label space is detected to be {target_column.estimated_n_classes}. Enabling extreme classification for the task."
            )

    @staticmethod
    def get_concrete_types(
        target: column_detector.CategoricalColumn,
        input_columns: typing.Dict[str, column_detector.Column],
    ):

        if target is None:
            raise Exception(
                f"Could not convert the specified target column into a valid categorical data type for TabularClassification."
            )

        data_types = {}
        data_types[target.column_name] = target

        for col in input_columns:
            data_types[col] = input_columns[col]

        return data_types


class RegressionTemplate(ModelBuilderTemplate):
    name = "regression"
    keywords = set(["regression"])
    description = "used to train a model for regression tasks and the output space is real numbers. can take arbitrary inputs like text numbers datetime etc."
    requires_explicit_instantiation = False

    def __init__(self, target_column_name, dataframe) -> None:
        self.target_column_name = target_column_name
        self.df = dataframe

        target_column = self.get_target_column(
            self.target_column_name,
            self.df,
            casting_function=column_detector.cast_to_numerical,
        )
        input_columns = get_input_columns(self.target_column_name, self.df)

        self.concrete_types = self.get_concrete_types(target_column, input_columns)

        self.bolt_data_types = {}

        for column_name, column in self.concrete_types.items():
            self.bolt_data_types[column_name] = column.to_bolt()

    @staticmethod
    def get_concrete_types(
        target: column_detector.CategoricalColumn,
        input_columns: typing.Dict[str, column_detector.Column],
    ) -> typing.Dict[str, column_detector.Column]:

        if target is None:
            raise Exception(
                f"Could not convert the specified target column into a valid numerical data type for Regression."
            )

        data_types = {}
        data_types[target.column_name] = target

        for col in input_columns:
            data_types[col] = input_columns[col]

        return data_types


class TokenClassificationTemplate(ModelBuilderTemplate):
    name = "ner"
    keywords = set(
        ["named entity recognition", "ner", "pii", "pii redaction", "llm firewall"]
    )
    description = "used to train a token classification model. used to assign a label to each token in the sentence. can be used for ner pii etc. input is space seperated text tokens and output is space seperated labels."

    requires_explict_instantiation = False

    def __init__(self, target_column_name, dataframe):
        self.target_column_name = target_column_name
        self.df = dataframe

        target_column = self.get_target_column(
            self.target_column_name,
            self.df,
            casting_function=column_detector.cast_to_categorical,
        )
        input_columns = get_input_columns(self.target_column_name, self.df)

        try:
            self.concrete_types = self.get_concrete_types(
                target_column, input_columns, self.df
            )
        except Exception as ex:
            message = ex.__str__()

            message = (
                message
                + "\n"
                + self.model_initialization_typehint(target_column.column_name)
            )

            raise Exception(message) from None

        self.bolt_data_types = {}

        for column_name, column in self.concrete_types.items():
            self.bolt_data_types[column_name] = column.to_bolt()

        if len(self.target_column.named_tags) > 250:
            raise Exception(
                f"Unexpected Number of Unique Tags Detected in the column {len(self.target_column.named_tags)} for Token Classification. Ensure that the column is correct"
            )

    @staticmethod
    def model_initialization_typehint(target_column_name: column_detector.TokenTags):
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
        target: column_detector.CategoricalColumn,
        input_columns: typing.Dict[str, column_detector.Column],
        dataframe: pd.DataFrame,
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

        token_column_candidates = get_token_candidates_for_token_classification(
            target, input_columns
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


class QueryReformulationTemplate(ModelBuilderTemplate):
    name = "query_reformulation"
    keywords = set(["reformulate", "query reformulation", "rephrase"])
    description = "used to train a model for query reformulation. pass in an input text and the output is also a text but reformulated. can be used for modifying grammatical errors in queries and other related tasks. can train in both supervised and unsupervised settings."
    requires_explict_instantiation = False

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

    def __init__(self, target_column_name, dataframe):
        self.target_column_name = target_column_name
        self.df = dataframe

        target_column = self.get_target_column(
            self.target_column_name,
            self.df,
            casting_function=column_detector.cast_to_categorical,
        )
        input_columns = get_input_columns(self.target_column_name, self.df)

        try:
            self.concrete_types = self.get_concrete_types(
                target_column, input_columns, self.df
            )
        except Exception as ex:
            message = ex.__str__()

            message = (
                message
                + "\n"
                + self.model_initialization_typehint(target_column.column_name)
            )

            raise Exception(message) from None

        self.bolt_data_types = {}

        for column_name, column in self.concrete_types.items():
            self.bolt_data_types[column_name] = column.to_bolt()

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

        token_column_candidates = get_source_column_for_query_reformulation(
            target, input_columns
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


class RecurrentClassifierTemplate(ModelBuilderTemplate):
    name = "rnn"
    keywords = set(["recurrence", "rnn", "sequential", "sequence"])
    description = "used to train an rnn model. when you want to predict sequences. the output is a sequence of categories which can be both string and integer format. use cases : time series predictions, directions while navigating."
    requires_explict_instantiation = False

    def __init__(self, target_column_name, dataframe) -> None:
        self.target_column_name = target_column_name
        self.df = dataframe

        target_column = self.get_target_column(
            self.target_column_name,
            self.df,
            casting_function=column_detector.cast_to_categorical,
        )
        input_columns = get_input_columns(self.target_column_name, self.df)

        self.concrete_types = self.get_concrete_types(
            target_column, input_columns, self.df
        )

        self.bolt_data_types = {}

        for column_name, column in self.concrete_types.items():
            if column_name == target_column.column_name:
                self.bolt_data_types[column_name] = column.to_bolt()
            else:
                self.bolt_data_types[column_name] = column.to_bolt()

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


class GraphClassificationTemplate(ModelBuilderTemplate):
    name = "graph_classification"
    keywords = set(["neighbours", "graph classification", "graph classifier"])
    description = "used for training models to perform classification tasks on graph-structured data. suitable for problems where the input data is represented as graphs, consisting of nodes and edges, and the goal is to predict a label for individual nodes. fraud review detection, social network entity classification, etc."

    requires_explicit_instantiation = True

    def __init__(self, target_column_name: str, dataframe: pd.DataFrame):
        raise Exception(
            "Graph Classifier Initialization is not yet supported by UDT Builder. Refer to https://github.com/ThirdAILabs/Demos/blob/main/universal_deep_transformer/graph_neural_networks/GraphNodeClassification.ipynb on how to initialize a Graph Classifier."
        )


supported_templates = [
    TabularClassificationTemplate,
    RegressionTemplate,
    TokenClassificationTemplate,
    QueryReformulationTemplate,
    RecurrentClassifierTemplate,
    GraphClassificationTemplate,
]
