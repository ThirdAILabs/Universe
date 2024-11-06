import logging
from typing import Any, Dict, Iterable, List, Optional, Union

import pandas as pd

from ..core.documents import Document
from ..core.types import MetadataType, NewChunkBatch, metadata_type_to_pandas_type
from .utils import join_metadata, series_from_value


def infer_text_columns(df: pd.DataFrame):
    return [column for column in df.columns if df[column].dtype == "object"]


def infer_pandas_column_types(
    data_iter: Iterable[pd.DataFrame], metadata_columns: Dict[str, Optional[str]]
):
    inferred_pandas_types = []
    for df in data_iter:
        for col, type in metadata_columns.items():

            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in the CSV.")

            if type:
                try:
                    # In ndb CSVs, we currently don't support multiple values in a column entry for a chunk.
                    # We will interpret a list value in a CSV as a string.
                    metadata_type = MetadataType(type)
                except ValueError:
                    allowed_types = ", ".join([dt.value for dt in MetadataType])
                    raise ValueError(
                        f"Data type '{type}' used for column '{col}' is not supported. "
                        f"Allowed types are: {allowed_types}"
                    )

                pandas_type = metadata_type_to_pandas_type[metadata_type]
                try:
                    df[col] = df[col].astype(pandas_type)
                except Exception as e:
                    raise ValueError(f"Cannot cast column '{col}' to type '{type}: {e}")

        inferred_pandas_types.append(df.dtypes)

    inferred_pandas_types_map = pd.DataFrame(inferred_pandas_types).max().to_dict()
    return inferred_pandas_types_map


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
        text_columns: Optional[List[str]] = None,
        keyword_columns: Optional[List[str]] = None,
        metadata_columns: Optional[Union[List[str], Dict[str, Optional[str]]]] = None,
        doc_metadata: Optional[Dict[str, Any]] = None,
        max_rows: int = 10_000_000,
        doc_id: Optional[str] = None,
        display_path: Optional[str] = None,
    ):
        super().__init__(doc_id=doc_id, doc_metadata=doc_metadata)

        self.path = path
        self.text_columns = text_columns
        self.keyword_columns = keyword_columns
        self.metadata_columns = metadata_columns
        self.max_rows = max_rows
        self.display_path = display_path

    def chunks(self) -> Iterable[NewChunkBatch]:
        data_iter = pd.read_csv(self.path, chunksize=self.max_rows)

        if self.metadata_columns and isinstance(self.metadata_columns, dict):

            inferred_pandas_types_map = infer_pandas_column_types(
                data_iter, self.metadata_columns
            )

            data_iter = pd.read_csv(
                self.path, chunksize=self.max_rows, dtype=inferred_pandas_types_map
            )

        for df in data_iter:
            if len(df) == 0:
                logging.warning(f"Inserting empty csv file {self.path} into NeuralDB.")
                continue

            df.reset_index(drop=True, inplace=True)

            if not self.text_columns and not self.keyword_columns:
                self.text_columns = infer_text_columns(df)
                self.keyword_columns = []
                if not self.text_columns:
                    raise Exception(f"No text columns found in {self.path}.")
            elif self.text_columns is None:
                self.text_columns = []
            elif self.keyword_columns is None:
                self.keyword_columns = []

            text = concat_str_columns(df, self.text_columns)

            keywords = concat_str_columns(df, self.keyword_columns)

            non_metadata_columns = [
                col
                for col in self.text_columns + self.keyword_columns
                if not (self.metadata_columns and col in self.metadata_columns)
            ]
            chunk_metadata = df.drop(non_metadata_columns, axis=1)
            metadata = join_metadata(
                n_rows=len(text),
                chunk_metadata=chunk_metadata,
                doc_metadata=self.doc_metadata,
            )

            yield NewChunkBatch(
                text=text,
                keywords=keywords,
                metadata=metadata,
                document=series_from_value(self.display_path or self.path, n=len(text)),
            )
