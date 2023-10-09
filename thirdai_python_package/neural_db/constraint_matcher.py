from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Generic, Iterable, List, Optional, Set, TypeVar
import pandas as pd

from sortedcontainers import SortedDict

ItemT = TypeVar("ItemT")


ItemConstraintIndex = SortedDict


class Filter(Generic[ItemT]):
    def filter(self, value_to_items: ItemConstraintIndex) -> Set[ItemT]:
        raise NotImplementedError()

    def filter_df_column(self, df: pd.DataFrame, column_name: str):
        raise NotImplementedError()


class AnyOf(Filter[ItemT]):
    def __init__(self, values: Iterable[Any]):
        self.values = values

    def filter(self, value_to_items: ItemConstraintIndex) -> Set[ItemT]:
        matches = set()
        for value in self.values:
            if value in value_to_items:
                matches = matches.union(value_to_items[value])
        return matches

    def filter_df_column(self, df: pd.DataFrame, column_name: str):
        return df[df[column_name].isin(self.values)]


class EqualTo(AnyOf[ItemT]):
    def __init__(self, value: Any):
        super().__init__([value])


class InRange(Filter[ItemT]):
    def __init__(
        self, minimum: Any, maximum: Any, inclusive_min=True, inclusive_max=True
    ):
        if minimum is None and maximum is None:
            raise ValueError("InRange cannot accept None for both minimum and maximum.")

        self.min = minimum
        self.max = maximum
        self.inclusive = (inclusive_min, inclusive_max)

    def filter(self, value_to_items: ItemConstraintIndex) -> Set[ItemT]:
        values = value_to_items.irange(self.min, self.max, self.inclusive)
        return AnyOf(values).filter(value_to_items)

    def filter_df_column(self, df: pd.DataFrame, column_name: str):
        left_inclusive, right_inclusive = self.inclusive
        if self.min is None and right_inclusive:
            return df[df[column_name].le(self.max)]
        elif self.min is None:
            return df[df[column_name].lt(self.max)]
        elif self.max is None and left_inclusive:
            return df[df[column_name].ge(self.min)]
        elif self.max is None:
            return df[df[column_name].gt(self.min)]

        if left_inclusive and right_inclusive:
            inclusive = "both"
        elif left_inclusive:
            inclusive = "left"
        elif right_inclusive:
            inclusive = "right"
        else:
            inclusive = "neither"

        return df[df[column_name].between(self.min, self.max, inclusive=inclusive)]


class GreaterThan(InRange[ItemT]):
    def __init__(self, minimum: Any, include_equal=False):
        super().__init__(minimum, maximum=None, inclusive_min=include_equal)


class LessThan(InRange[ItemT]):
    def __init__(self, maximum: Any, include_equal=False):
        super().__init__(minimum=None, maximum=maximum, inclusive_max=include_equal)


class ConstraintValue:
    def __init__(self, value: Any = None, is_any: bool = False):
        if is_any and value is not None:
            raise RuntimeError(
                "ConstraintValue cannot accept non-None value and is_any=True at the same time."
            )
        self._value = value
        self._is_any = is_any

    def any(self):
        return self._is_any

    def value(self):
        return self._value


class ConstraintIndex(Generic[ItemT]):
    def __init__(self):
        self._any_value = set()
        self._match_value = ItemConstraintIndex()

    def match(self, filterer: Filter) -> Set[ItemT]:
        return self._any_value.union(filterer.filter(self._match_value))

    def index(self, item: ItemT, constraint_value: ConstraintValue) -> None:
        if constraint_value.any():
            self._any_value.add(item)
        else:
            value = constraint_value.value()
            if not value in self._match_value:
                self._match_value[value] = set()
            self._match_value[constraint_value.value()].add(item)


class ConstraintMatcher(Generic[ItemT]):
    def __init__(self):
        self._all_items = set()
        self._item_constraints = {}

    def match(self, filters: Dict[str, Filter]) -> Set[ItemT]:
        matches = self._all_items

        for key, filterer in filters.items():
            if key not in self._item_constraints:
                return set()
            matches = matches.intersection(self._item_constraints[key].match(filterer))

        return matches

    def index(self, item: ItemT, constraints: Dict[str, ConstraintValue]) -> None:
        for key, constraint_value in constraints.items():
            self._all_items.add(item)
            if key not in self._item_constraints:
                self._item_constraints[key] = ConstraintIndex[ItemT]()
            self._item_constraints[key].index(item, constraint_value)


def to_filters(constraints: Dict[str, Any]):
    return {
        key: value if isinstance(value, Filter) else EqualTo(value)
        for key, value in constraints.items()
    }
