from typing import Any

import pandas as pd


def series_from_value(value: Any, n: int) -> pd.Series:
    return pd.Series(value).repeat(len(n)).reset_index(drop=True)
