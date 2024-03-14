from abc import ABC, abstractmethod

from sqlalchemy import Table


class Constraint(ABC):
    @abstractmethod
    def sql_condition(self, column_name: str, table: Table):
        pass


class EqualTo(Constraint):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def sql_condition(self, column_name: str, table: Table):
        return table.c[column_name] == self.value


class AnyOf(Constraint):
    def __init__(self, values):
        super().__init__()
        self.values = values

    def sql_condition(self, column_name: str, table: Table):
        return table.c[column_name].in_(self.values)


class GreaterThan(Constraint):
    def __init__(self, value, inclusive=True):
        super().__init__()
        self.value = value
        self.inclusive = inclusive

    def sql_condition(self, column_name: str, table: Table):
        if self.inclusive:
            return table.c[column_name] >= self.value
        return table.c[column_name] > self.value


class LessThan(Constraint):
    def __init__(self, value, inclusive=True):
        super().__init__()
        self.value = value
        self.inclusive = inclusive

    def sql_condition(self, column_name: str, table: Table):
        if self.inclusive:
            return table.c[column_name] <= self.value
        return table.c[column_name] < self.value
