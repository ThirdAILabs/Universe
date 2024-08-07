from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from ..core.documents import Document
from ..core.types import NewChunkBatch
from .utils import join_metadata, series_from_value


def is_text_column(column: pd.Series):
    return (
        column.dtype == "object"
        and column[:200].map(lambda x: isinstance(x, str)).all()
    )


def infer_text_columns(df: pd.DataFrame):
    return [column for column in df.columns if is_text_column(df[column])]


def concat_str_columns(df: pd.DataFrame, columns: List[str]):
    if len(columns) == 0:
        return series_from_value(value="", n=len(df))

    output = df[columns[0]].fillna("")

    for col in columns[1:]:
        output = output + " " + df[col].fillna("")

    return output


class CSV(Document):
    def __init__(
        self,
        path,
        text_columns=[],
        keyword_columns=[],
        doc_metadata=None,
        max_rows=10_000_000,
        doc_id: Optional[str] = None,
    ):
        super().__init__(doc_id=doc_id)

        self.path = path
        self.text_columns = text_columns
        self.keyword_columns = keyword_columns
        self.doc_metadata = doc_metadata
        self.max_rows = max_rows

    def chunks(self) -> Iterable[NewChunkBatch]:
        data_iter = pd.read_csv(self.path, chunksize=self.max_rows)

        for df in data_iter:
            df.reset_index(drop=True, inplace=True)

            if len(self.text_columns) + len(self.keyword_columns) == 0:
                self.text_columns = infer_text_columns(df)

            text = concat_str_columns(df, self.text_columns)
            keywords = concat_str_columns(df, self.keyword_columns)

            chunk_metadata = df.drop(self.text_columns + self.keyword_columns, axis=1)
            metadata = join_metadata(
                n_rows=len(text),
                chunk_metadata=chunk_metadata,
                doc_metadata=self.doc_metadata,
            )

            yield NewChunkBatch(
                text=text,
                keywords=keywords,
                metadata=metadata,
                document=series_from_value(self.path, n=len(text)),
            )
