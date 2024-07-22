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
    text: pt.Series[str]
    keywords: pt.Series[str]
    metadata: Optional[pt.DataFrame]
    document: pt.Series[str]

    def __post_init__(self):
        assert isinstance(self.text, pd.Series)
        assert isinstance(self.keywords, pd.Series)
        assert isinstance(self.metadata, pd.DataFrame) or self.metadata is None
        assert isinstance(self.document, pd.Series)

        fields_to_check = [self.text, self.keywords, self.document]

        if not self.metadata is None:
            fields_to_check.append(self.metadata)

        lengths = set(len(x) for x in fields_to_check)
        if len(lengths) != 1:
            raise ValueError("Must have fields of the same length in NewChunkBatch.")

        if len(self.text) == 0:
            raise ValueError("Cannot create empty NewChunkBatch.")

    def __len__(self):
        return len(self.text)

    def to_df(self):
        return pd.DataFrame(
            {
                "text": self.text,
                "keywords": self.keywords,
                "document": self.document,
            }
        )


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

    def __len__(self):
        return len(self.text)

    def to_df(self):
        return pd.DataFrame(self.__dict__)


@dataclass
class SupervisedSample:
    query: str
    chunk_id: List[ChunkId]


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


@dataclass
class InsertedDocMetadata:
    doc_id: str
    doc_version: int
    chunk_ids: List[ChunkId]
