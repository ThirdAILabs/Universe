import uuid
from abc import ABC, abstractmethod
from typing import Iterable

from .types import NewChunkBatch


class Document(ABC):
    def __init__(self, doc_id):
        self._doc_id = doc_id or str(uuid.uuid4())

    @abstractmethod
    def chunks(self) -> Iterable[NewChunkBatch]:
        raise NotImplementedError

    def doc_id(self):
        return self._doc_id
