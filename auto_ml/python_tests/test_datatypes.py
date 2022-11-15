import numpy as np
import pandas as pd
import pytest
from thirdai.bolt import types

pytestmark = [pytest.mark.unit]

def test_datatypes_to_string():
    assert (str(types.categorical(n_unique_classes=1)) == '{"type": "text"}')
