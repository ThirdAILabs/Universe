from abc import ABC, abstractmethod
from typing import Iterable

from .types import NewChunkBatch


class Document(ABC):
    @abstractmethod
    def chunks(self) -> Iterable[NewChunkBatch]:
        raise NotImplementedError
