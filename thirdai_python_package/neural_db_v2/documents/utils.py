from typing import Any, Optional

import numpy as np
import pandas as pd


def series_from_value(value: Any, n: int) -> pd.Series:
    return pd.Series(np.full(n, value))


def join_metadata(
    n_rows,
    chunk_metadata: Optional[pd.DataFrame] = None,
    doc_metadata: Optional[dict] = None,
) -> Optional[pd.DataFrame]:
    if isinstance(chunk_metadata, pd.DataFrame) and len(chunk_metadata) == 0:
        chunk_metadata = None

    if doc_metadata is not None:
        doc_metadata = pd.DataFrame.from_records([doc_metadata] * n_rows)

    if chunk_metadata is not None and len(chunk_metadata) != n_rows:
        raise ValueError("Length of chunk metadata must match number of chunks.")

    if doc_metadata is not None and chunk_metadata is not None:
        return pd.concat([chunk_metadata, doc_metadata], axis=1)
    if doc_metadata is not None:
        return doc_metadata
    if chunk_metadata is not None:
        return chunk_metadata
    return None
