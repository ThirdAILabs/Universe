from typing import Iterable, List

import pandas as pd

from ..core.documents import Document
from ..core.types import NewChunkBatch
from .utils import series_from_value


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

    output = df[columns[0]]

    for col in columns[1:]:
        output += " " + df[col]

    return output


class CSV(Document):
    def __init__(
        self,
        path,
        text_columns=[],
        keyword_columns=[],
        custom_id_column=None,
        metadata=None,
    ):
        super().__init__()

        self.path = path
        self.text_columns = text_columns
        self.keyword_columns = keyword_columns
        self.custom_id_column = custom_id_column
        self.metadata = metadata

    def chunks(self) -> Iterable[NewChunkBatch]:
        df = pd.read_csv(self.path)

        custom_id = df[self.custom_id_column] if self.custom_id_column else None
        df.drop(self.custom_id_column, axis=1, inplace=True)

        if len(self.text_columns) + len(self.keyword_columns) == 0:
            self.text_columns = infer_text_columns(df)

        text = concat_str_columns(df, self.text_columns)
        keywords = concat_str_columns(df, self.keyword_columns)

        metadata = df.drop(self.text_columns + self.keyword_columns, axis=1)

        if self.metadata is not None:
            metadata = pd.concat(
                [metadata, pd.DataFrame.from_records([self.metadata] * len(text))],
                axis=1,
            )

        metadata = metadata if len(metadata.columns) > 0 else None

        return [
            NewChunkBatch(
                custom_id=custom_id,
                text=text,
                keywords=keywords,
                metadata=metadata,
                document=series_from_value(self.path, n=len(text)),
            )
        ]
