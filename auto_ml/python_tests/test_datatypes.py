import numpy as np
import pandas as pd
import pytest
from thirdai.bolt import types

pytestmark = [pytest.mark.unit]


def test_categorical_datatypes_to_string():
    assert str(types.categorical()) == '{"type": "categorical"}'
    assert (
        str(types.categorical(delimiter=":"))
        == '{"type": "categorical", "delimiter": ":"}'
    )
    assert (
        str(types.categorical(delimiter="-"))
        == '{"type": "categorical", "delimiter": "-"}'
    )


def test_text_datatype_to_string():
    assert str(types.text()) == '{"type": "text"}'


def test_date_datatype_to_string():
    assert str(types.date()) == '{"type": "date"}'


def test_numerical_datatype_to_string():
    assert (
        str(types.numerical(range=(0, 1)))
        == '{"type": "numerical", "range": [0, 1], "granularity": "m"}'
    )
    assert (
        str(types.numerical(range=(-10, 50), granularity="w"))
        == '{"type": "numerical", "range": [-10, 50], "granularity": "w"}'
    )


# If a list of datatypes to string works, then so will a map. We test a list
# because the order of the items in the str() representation is deterministic.
def test_datatype_list_to_string():
    test_list = [
        types.numerical(range=(0, 1)),
        types.text(),
        types.categorical(),
        types.date(),
    ]
    assert (
        str(test_list)
        == '[{"type": "numerical", "range": [0, 1], "granularity": "m"}, {"type": "text"}, {"type": "categorical"}, {"type": "date"}]'
    )


def test_token_tag_with_space():
    tags = ["email", "credit card"]
    default_tag = "O"

    with pytest.raises(
        ValueError, match="Tags with spaces are not allowed. Found tag: 'credit card'"
    ):
        token_tags = types.token_tags(tags=tags, default_tag=default_tag)
