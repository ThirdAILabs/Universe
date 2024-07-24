import uuid
from abc import ABC, abstractmethod
from typing import Iterable, Optional

from .types import NewChunkBatch


class Document(ABC):
    def __init__(self, doc_id: Optional[str]):
        self._doc_id = doc_id or str(uuid.uuid4())

    @abstractmethod
    def chunks(self) -> Iterable[NewChunkBatch]:
        raise NotImplementedError

    def doc_id(self) -> str:
        return self._doc_id

    def __iter__(self) -> Iterable[NewChunkBatch]:
        return iter(self.chunks())
