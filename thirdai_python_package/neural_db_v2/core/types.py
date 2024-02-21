from dataclasses import dataclass
from typing import Union

# We typedef doc ID to anticipate switching over to string IDs
ChunkId = int


@dataclass
class NewChunk:
    """A chunk that has not been assigned a unique ID."""

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


@dataclass
class Chunk(NewChunk):
    """A chunk that has been assigned a unique ID."""

    # A unique identifier assigned by a chunk store.
    chunk_id: ChunkId


@dataclass
class CustomIdSupervisedSample:
    query: str
    custom_id: Union[str, int]


@dataclass
class SupervisedSample:
    query: str
    chunk_id: int
