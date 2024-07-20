from abc import ABC, abstractmethod
from typing import Iterable, List, Union

import pandas as pd

from .types import ChunkId, SupervisedBatch


class SupervisedDataset(ABC):
    @abstractmethod
    def samples(self) -> Iterable[SupervisedBatch]:
        raise NotImplementedError

    def supervised_samples(
        self, queries: List[str], ids: List[List[ChunkId]]
    ) -> SupervisedBatch:
        return SupervisedBatch(
            query=queries,
            chunk_id=ids,
        )
