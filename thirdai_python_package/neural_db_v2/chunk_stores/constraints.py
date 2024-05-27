from abc import ABC, abstractmethod

import pandas as pd
from sqlalchemy import Table


class Constraint(ABC):
    @abstractmethod
    def sql_condition(self, column_name: str, table: Table):
        raise NotImplementedError

    @abstractmethod
    def pd_filter(self, column_name: str, df: pd.DataFrame):
        raise NotImplementedError


class EqualTo(Constraint):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def sql_condition(self, column_name: str, table: Table):
        return table.c[column_name] == self.value

    def pd_filter(self, column_name: str, df: pd.DataFrame):
        return df[column_name] == self.value


class AnyOf(Constraint):
    def __init__(self, values):
        super().__init__()
        self.values = values

    def sql_condition(self, column_name: str, table: Table):
        return table.c[column_name].in_(self.values)

    def pd_filter(self, column_name: str, df: pd.DataFrame):
        return df[column_name].isin(self.values)


class GreaterThan(Constraint):
    def __init__(self, value, inclusive=True):
        super().__init__()
        self.value = value
        self.inclusive = inclusive

    def sql_condition(self, column_name: str, table: Table):
        if self.inclusive:
            return table.c[column_name] >= self.value
        return table.c[column_name] > self.value

    def pd_filter(self, column_name: str, df: pd.DataFrame):
        if self.inclusive:
            return df[column_name] >= self.value
        return df[column_name] > self.value


class LessThan(Constraint):
    def __init__(self, value, inclusive=True):
        super().__init__()
        self.value = value
        self.inclusive = inclusive

    def sql_condition(self, column_name: str, table: Table):
        if self.inclusive:
            return table.c[column_name] <= self.value
        return table.c[column_name] < self.value

    def pd_filter(self, column_name: str, df: pd.DataFrame):
        if self.inclusive:
            return df[column_name] <= self.value
        return df[column_name] < self.value
