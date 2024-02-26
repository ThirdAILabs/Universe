from abc import ABC, abstractmethod
from typing import Iterable

from core.types import NewChunkBatch


class Document(ABC):
    @abstractmethod
    def chunks(self) -> Iterable[NewChunkBatch]:
        pass
