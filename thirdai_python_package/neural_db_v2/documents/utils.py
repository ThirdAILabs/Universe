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
    if chunk_metadata is not None and len(chunk_metadata) != n_rows:
        raise ValueError("Length of chunk metadata must match number of chunks.")

    if chunk_metadata is None:
        chunk_metadata = pd.DataFrame()

    if doc_metadata:
        doc_metadata = pd.DataFrame.from_records([doc_metadata] * n_rows)
    else:
        doc_metadata = pd.DataFrame()

    metadata = pd.concat([chunk_metadata, doc_metadata], axis=1)

    return metadata if not metadata.empty else None
