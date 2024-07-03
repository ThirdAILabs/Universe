import typing
from abc import abstractclassmethod
from dataclasses import dataclass

import thirdai._thirdai.bolt as bolt


@dataclass
class Column:
    name: str

    @abstractclassmethod
    def to_bolt(self, **kwargs):
        pass


"""
Target column for each task can be broadly categorized into two categories namely, Numerical Column and Categorical Column.

Example : 
1. Text columns is just a Categorical Column with a space delimiter and higher number of unique tokens.
2. TokenTags is just a Categorical Column with a space delimiter and a corresponding tokens column.
3. Sequence Column is a Categorical Column with a custom delimiter.

We can further cast the Categorical Column into relevant types like TokenTags depending upon the detected task. 
"""


@dataclass
class CategoricalColumn(Column):
    token_type: typing.Union[str, int, float]
    number_tokens_per_row: float
    unique_tokens_per_row: float
    estimated_n_classes: int

    delimiter: typing.Optional[str] = None

    def to_bolt(self, is_target_type=False, **kwargs):
        # if the column is a target column, n_classes is required
        return bolt.types.categorical(
            type=self.token_type.__name__,
            delimiter=self.delimiter,
            n_classes=self.estimated_n_classes if is_target_type else None,
        )


@dataclass
class NumericalColumn(Column):
    minimum: float
    maximum: float

    def to_bolt(self, **kwargs):
        return bolt.types.numerical((self.minimum, self.maximum))


@dataclass
class DateTimeColumn(Column):
    def to_bolt(self, **kwargs):
        return bolt.types.date()


@dataclass
class TextColumn(Column):
    def to_bolt(self, **kwargs):
        return bolt.types.text()


@dataclass
class TokenTags(Column):
    default_tag: str
    named_tags: typing.List[str]

    def to_bolt(self, **kwargs):
        return bolt.types.token_tags(tags=self.named_tags, default_tag=self.default_tag)


@dataclass
class SequenceType(Column):
    delimiter: str
    estimated_n_classes: int = None
    max_length: int = None

    def __post_init__(self):
        if self.delimiter is None:
            raise Exception(
                "The delimiter for a sequence type column is None. Ensure that the entries in a column are valid sequences."
            )

    def to_bolt(self, is_target_type=False, **kwargs):
        return bolt.types.sequence(
            delimiter=self.delimiter,
            n_classes=self.estimated_n_classes if is_target_type else None,
            max_length=self.max_length if is_target_type else None,
        )
