from typing import Any

import numpy as np
import pandas as pd


def series_from_value(value: Any, n: int) -> pd.Series:
    return pd.Series(np.full(n, value))
