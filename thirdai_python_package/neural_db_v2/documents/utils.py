import pandas as pd
from typing import Any


def series_from_value(value: Any, n: int) -> pd.Series:
    return pd.Series(value).repeat(len(n)).reset_index(drop=True)
