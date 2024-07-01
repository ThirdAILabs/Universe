from __future__ import annotations

import typing

import thirdai._thirdai.bolt as bolt
import pandas as pd

from .model_templates import UDTDataTemplate
from ..column_inferencing import column_detector


class ModelBuilder:

    def __init__(
        self,
        dataframe: pd.DataFrame,
        target_column: column_detector.Column,
        input_columns: typing.Dict[str, column_detector.Column],
        template: UDTDataTemplate,
    ):
        self.df = dataframe
        self.template = template

        self.target_column, self.input_columns = self.template.get_concrete_types(
            target_column, input_columns, self.df
        )

        self.bolt_data_types = {}

        for column_name, column in self.input_columns.items():
            self.bolt_data_types[column_name] = column.to_bolt()

        if isinstance(
            self.target_column,
            column_detector.CategoricalColumn,
        ):
            self.bolt_data_types[self.target_column_name] = self.target_column.to_bolt(
                is_target_type=True
            )

    @property
    def target_column_name(self):
        return self.target_column.name

    def build(self):
        extreme_classification = False

        if isinstance(self.target_column, column_detector.CategoricalColumn):
            if self.target_column.estimated_n_classes > 100_000:
                extreme_classification = True

        self.model = bolt.UniversalDeepTransformer(
            data_types=self.bolt_data_types,
            target=self.target_column_name,
            extreme_classification=extreme_classification,
        )

        return self.model

    @staticmethod
    def from_raw_types(target_column_name, dataframe, template: UDTDataTemplate):
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

        return ModelBuilder(dataframe, target_column, input_columns, template)

    @property
    def task(self):
        return self.template.task
