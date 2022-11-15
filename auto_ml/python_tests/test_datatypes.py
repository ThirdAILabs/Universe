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
