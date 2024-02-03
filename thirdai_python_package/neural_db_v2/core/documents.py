from abc import ABC, abstractmethod
from typing import Iterable
from core.types import NewChunk


class Document(ABC):
    @abstractmethod
    def chunks(self) -> Iterable[NewChunk]:
        pass
