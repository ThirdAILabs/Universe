from __future__ import annotations
from typing import List, Set, Any, TypeVar, Generic, Dict, Optional, Set, Iterable
from collections import defaultdict
from sortedcontainers import SortedDict


ItemT = TypeVar("ItemT")

# TODO(Geordie): Support range queries


ValueItemIndex = SortedDict


class Filter(Generic[ItemT]):
    def filter(self, value_to_items: ValueItemIndex) -> Set[ItemT]:
        raise NotImplementedError()


class AnyOf(Filter[ItemT]):
    def __init__(self, values: Iterable[Any]):
        self.values = values

    def filter(self, value_to_items: ValueItemIndex) -> Set[ItemT]:
        matches = set()
        for value in self.values:
            if value in value_to_items:
                matches = matches.union(value_to_items[value])
        return matches


class EqualTo(Filter[ItemT]):
    def __init__(self, value: Any):
        self.any_of = AnyOf([value])

    def filter(self, value_to_items: ValueItemIndex) -> Set[ItemT]:
        return self.any_of.filter(value_to_items)


class InRange(Filter[ItemT]):
    def __init__(
        self, minimum: Any, maximum: Any, inclusive_min=True, inclusive_max=True
    ):
        self.min = minimum
        self.max = maximum
        self.inclusive = (inclusive_min, inclusive_max)

    def filter(self, value_to_items: ValueItemIndex) -> Set[ItemT]:
        values = value_to_items.irange(self.min, self.max, self.inclusive)
        return AnyOf(values).filter(value_to_items)


class GreaterThan(Filter[ItemT]):
    def __init__(self, minimum: Any, include_equal=False):
        self.in_range = InRange(minimum, maximum=None, inclusive_min=include_equal)

    def filter(self, value_to_items: ValueItemIndex) -> Set[ItemT]:
        return self.in_range.filter(value_to_items)


class LessThan(Filter[ItemT]):
    def __init__(self, maximum: Any, include_equal=False):
        self.in_range = InRange(
            minimum=None, maximum=maximum, inclusive_max=include_equal
        )

    def filter(self, value_to_items: ValueItemIndex) -> Set[ItemT]:
        return self.in_range.filter(value_to_items)


class ConstraintValue:
    def __init__(self, value: Optional[Any]):
        self._value = value

    def any(self):
        return self._value is None

    def value(self):
        return self._value


class ConstraintIndex(Generic[ItemT]):
    def __init__(self):
        self._any_value = Set[ItemT]()
        self._match_value = ValueItemIndex()

    def match(self, filterer: Filter) -> Set[ItemT]:
        return self._any_value.union(filterer.filter(self._match_value))

    def index(self, item: ItemT, constraint_value: ConstraintValue) -> None:
        if constraint_value.any():
            self._any_value.add(item)
        else:
            self._match_value[constraint_value.value()].add(item)


class ConstraintMatcher(Generic[ItemT]):
    def __init__(self):
        self._all_items = set()
        self._item_constraints = defaultdict(lambda: ConstraintIndex[ItemT]())

    def match(self, filters: Dict[str, Filter]) -> Set[ItemT]:
        matches = self._all_items

        for key, filterer in filters.items():
            matches = matches.intersection(self._item_constraints[key].match(filterer))

        return matches

    def index(self, item: ItemT, constraints: Dict[str, ConstraintValue]) -> None:
        for key, constraint_value in constraints.items():
            self._all_items.add(item)
            self._item_constraints[key].index(item, constraint_value)


def to_filters(constraints: Dict[str, Any]):
    return {
        key: value if isinstance(value, Filter) else EqualTo(value)
        for key, value in constraints.items()
    }
