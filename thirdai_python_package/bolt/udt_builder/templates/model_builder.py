from __future__ import annotations
from abc import abstractclassmethod
import typing


from thirdai import bolt
import pandas as pd

from .model_templates import UDTDataTemplate
from ..column_inferencing import column_detector


class ModelBuilder:
    name = "default"
    keywords = set()
    description = "default model"

    @staticmethod
    def get_target_column(
        target_column_name, dataframe: pd.DataFrame, casting_function
    ):
        try:
            column = casting_function(target_column_name, dataframe[target_column_name])
            return column
        except:
            return None

    def __init__(
        self,
        target_column_name,
        dataframe,
        target_column,
        input_columns,
        template: UDTDataTemplate,
    ):
        self.target_column_name = target_column_name
        self.df = dataframe
        self.target_column = target_column
        self.input_columns = input_columns
        self.template = template

        self.concrete_types = self.template.get_concrete_types(
            self.target_column, self.input_columns, self.df
        )

        self.bolt_data_types = {}

        for column_name, column in self.concrete_types.items():
            if column_name == target_column.column_name and isinstance(
                self.concrete_types[target_column_name],
                column_detector.CategoricalColumn,
            ):
                self.bolt_data_types[column_name] = column.to_bolt(is_target_type=True)
            else:
                self.bolt_data_types[column_name] = column.to_bolt()

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

    @staticmethod
    def get_builder_from_raw_types(
        target_column_name, dataframe, template: UDTDataTemplate
    ):
        try:
            target_column = template.target_column_caster(
                target_column_name, dataframe[target_column_name]
            )
        except:
            target_column = None
            raise Exception(
                f"Could not convert the specified target column {target_column_name} into a valid datatype for {template.task}. Make sure that the target column name is correct and is a valid data type for the specified task."
            )

        input_columns = column_detector.get_input_columns(target_column_name, dataframe)

        return ModelBuilder(
            target_column_name, dataframe, target_column, input_columns, template
        )

    @property
    def task(self):
        return self.template.task
