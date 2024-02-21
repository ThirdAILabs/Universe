from dataclasses import dataclass
from typing import Union

# We typedef doc ID to anticipate switching over to string IDs
ChunkId = int


@dataclass
class NewChunk:
    user_assigned_id: Union[str, int, None]
    text: str
    keywords: str
    metadata: dict


@dataclass
class Chunk(NewChunk):
    chunk_id: ChunkId
