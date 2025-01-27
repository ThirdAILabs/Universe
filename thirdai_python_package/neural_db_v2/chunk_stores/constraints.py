from abc import ABC, abstractmethod

import pandas as pd
from sqlalchemy import Table, and_, or_


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
        return and_(table.c.key == column_name, table.c.value == self.value)

    def pd_filter(self, column_name: str, df: pd.DataFrame):
        return df[column_name] == self.value


class AnyOf(Constraint):
    def __init__(self, values):
        super().__init__()
        self.values = values

    def sql_condition(self, column_name: str, table: Table):
        return and_(table.c.key == column_name, table.c.value.in_(self.values))

    def pd_filter(self, column_name: str, df: pd.DataFrame):
        return df[column_name].isin(self.values)


# NoneOf's behavior is to return any chunk that contains any explicit value (not None)
# other than those in the 'values' parameter
class NoneOf(Constraint):
    def __init__(self, values):
        super().__init__()
        self.values = values

    def sql_condition(self, column_name: str, table: Table):
        return and_(table.c.key == column_name, ~table.c.value.in_(self.values))

    def pd_filter(self, column_name: str, df: pd.DataFrame):
        return ~df[column_name].isin(self.values) & pd.notna(df[column_name])


class GreaterThan(Constraint):
    def __init__(self, value, inclusive=True):
        super().__init__()
        self.value = value
        self.inclusive = inclusive

    def sql_condition(self, column_name: str, table: Table):
        comparison = (
            table.c.value >= self.value
            if self.inclusive
            else table.c.value > self.value
        )
        return and_(table.c.key == column_name, comparison)

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
        comparison = (
            table.c.value <= self.value
            if self.inclusive
            else table.c.value < self.value
        )
        return and_(table.c.key == column_name, comparison)

    def pd_filter(self, column_name: str, df: pd.DataFrame):
        if self.inclusive:
            return df[column_name] <= self.value
        return df[column_name] < self.value


class InRange(Constraint):
    def __init__(
        self,
        min_value,
        max_value,
        min_inclusive: bool = True,
        max_inclusive: bool = True,
    ):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value
        self.min_inclusive = min_inclusive
        self.max_inclusive = max_inclusive

    def sql_condition(self, column_name: str, table: Table):
        if self.min_inclusive:
            lower_condition = table.c.value >= self.min_value
        else:
            lower_condition = table.c.value > self.min_value

        if self.max_inclusive:
            upper_condition = table.c.value <= self.max_value
        else:
            upper_condition = table.c.value < self.max_value

        return and_(table.c.key == column_name, lower_condition, upper_condition)

    def pd_filter(self, column_name: str, df: pd.DataFrame):
        if self.min_inclusive:
            lower_condition = df[column_name] >= self.min_value
        else:
            lower_condition = df[column_name] > self.min_value

        if self.max_inclusive:
            upper_condition = df[column_name] <= self.max_value
        else:
            upper_condition = df[column_name] < self.max_value

        return lower_condition & upper_condition


class Substring(Constraint):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def sql_condition(self, column_name: str, table: Table):
        return and_(table.c.key == column_name, table.c.value.like(f"%{self.value}%"))

    def pd_filter(self, column_name: str, df: pd.DataFrame):
        if not pd.api.types.is_string_dtype(df[column_name]):
            return pd.Series([False] * len(df), index=df.index)
        return df[column_name].str.contains(self.value, na=False)
