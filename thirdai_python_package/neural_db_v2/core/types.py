from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from pandera import typing as pt
from thirdai import data

# We typedef doc ID to anticipate switching over to string IDs
ChunkId = int

Score = float


@dataclass
class Chunk:
    """A chunk that has been assigned a unique ID."""

    # An optional identifier supplied by the user.
    custom_id: Union[str, int, None]

    # The text content of the chunk, e.g. a paragraph.
    text: str

    # Keywords / strong signals.
    keywords: str

    # Arbitrary metadata related to the chunk.
    metadata: dict

    # Parent document name
    document: str

    # UUID for the document
    doc_id: str

    # Version of the document
    doc_version: int

    # A unique identifier assigned by a chunk store.
    chunk_id: ChunkId


"""Design choices for batch objects:
- Column oriented so we can efficiently convert it to a ColumnMap
- Pandas series instead of Columns.
  - Series can contain dictionaries, which is useful for the metadata field. 
  - Many libraries natively accept Series or Numpy arrays, which
    Series can easily convert into, so this is useful for when we implement
    chunk stores or retrievers using external libraries.
  - Series are easy to work with in Python, preventing the need to write 
    more bindings and tools for Columns.
- Store individual columns as named fields instead of storing a dataframe to
  prevent errors from column name typos.
- __getitem__ method to access individual rows for convenience.
"""


@dataclass
class NewChunkBatch:
    custom_id: Union[pt.Series[str], pt.Series[int], None]
    text: pt.Series[str]
    keywords: pt.Series[str]
    metadata: Optional[pt.DataFrame]
    document: pt.Series[str]

    def __post_init__(self):
        assert isinstance(self.custom_id, pd.Series) or self.custom_id is None
        assert isinstance(self.text, pd.Series)
        assert isinstance(self.keywords, pd.Series)
        assert isinstance(self.metadata, pd.DataFrame) or self.metadata is None
        assert isinstance(self.document, pd.Series)

        fields_to_check = [self.text, self.keywords, self.document]

        if not self.custom_id is None:
            fields_to_check.append(self.custom_id)

        if not self.metadata is None:
            fields_to_check.append(self.metadata)

        lengths = set(len(x) for x in fields_to_check)
        if len(lengths) != 1:
            raise ValueError("Must have fields of the same length in NewChunkBatch.")

        if len(self.text) == 0:
            raise ValueError("Cannot create empty NewChunkBatch.")

    def __len__(self):
        return len(self.text)


@dataclass
class VersionedNewChunkBatch(NewChunkBatch):
    doc_id: pd.Series[str]
    doc_version: pd.Series[int]

    def __post_init__(self):
        super().__post_init__()

        assert isinstance(self.doc_id, pd.Series)
        assert isinstance(self.doc_version, pd.Series)

        assert len(self.doc_id) == len(self.text)
        assert len(self.doc_version) == len(self.text)

    def to_df(self):
        columns = {
            "text": self.text,
            "keywords": self.keywords,
            "document": self.document,
            "doc_id": self.doc_id,
            "doc_version": self.doc_version,
        }
        if self.custom_id is not None:
            columns["custom_id"] = self.custom_id
        else:
            columns["custom_id"] = pd.Series(np.full(len(self.text), None))

        return pd.DataFrame(columns)


@dataclass
class ChunkBatch:
    chunk_id: pt.Series[ChunkId]
    text: pt.Series[str]
    keywords: pt.Series[str]

    def __post_init__(self):
        assert isinstance(self.chunk_id, pd.Series)
        assert isinstance(self.text, pd.Series)
        assert isinstance(self.keywords, pd.Series)

        self.chunk_id = self.chunk_id.reset_index(drop=True)
        self.text = self.text.reset_index(drop=True)
        self.keywords = self.keywords.reset_index(drop=True)

        if not (len(self.chunk_id) == len(self.text) == len(self.keywords)):
            raise ValueError("Must have fields of the same length in ChunkBatch.")

        if len(self.text) == 0:
            raise ValueError("Cannot create empty ChunkBatch.")

    def to_df(self):
        return pd.DataFrame(self.__dict__)


@dataclass
class CustomIdSupervisedSample:
    query: str
    custom_id: Union[List[str], List[int]]


@dataclass
class SupervisedSample:
    query: str
    chunk_id: List[ChunkId]


@dataclass
class CustomIdSupervisedBatch:
    query: pt.Series[str]
    custom_id: Union[pt.Series[List[str]], pt.Series[List[int]]]

    def __post_init__(self):
        assert isinstance(self.custom_id, pd.Series)
        assert isinstance(self.query, pd.Series)

        self.query = self.query.reset_index(drop=True)
        self.custom_id = self.custom_id.reset_index(drop=True)

        if len(self.query) != len(self.custom_id):
            raise ValueError(
                "Must have fields of the same length in CustomIdSupervisedBatch."
            )

        if len(self.query) == 0:
            raise ValueError("Cannot create empty CustomIdSupervisedBatch.")

    def __getitem__(self, i: int):
        return CustomIdSupervisedSample(
            query=self.query[i],
            custom_id=self.custom_id[i],
        )

    def to_df(self):
        return pd.DataFrame(self.__dict__)


@dataclass
class SupervisedBatch:
    query: pt.Series[str]
    chunk_id: pt.Series[List[ChunkId]]

    def __post_init__(self):
        assert isinstance(self.chunk_id, pd.Series)
        assert isinstance(self.query, pd.Series)

        self.query = self.query.reset_index(drop=True)
        self.chunk_id = self.chunk_id.reset_index(drop=True)

        if len(self.query) != len(self.chunk_id):
            raise ValueError("Must have fields of the same length in SupervisedBatch.")

        if len(self.query) == 0:
            raise ValueError("Cannot create empty SupervisedBatch.")

    def __getitem__(self, i: int):
        return SupervisedSample(
            query=self.query[i],
            chunk_id=self.chunk_id[i],
        )

    def to_df(self):
        return pd.DataFrame(self.__dict__)
