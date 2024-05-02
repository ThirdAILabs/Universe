from abc import ABC, abstractmethod
from typing import Iterable, List, Union

import pandas as pd

from .types import ChunkId, CustomIdSupervisedBatch, SupervisedBatch


class SupervisedDataset(ABC):
    @abstractmethod
    def samples(
        self,
    ) -> Union[Iterable[SupervisedBatch], Iterable[CustomIdSupervisedBatch]]:
        raise NotImplementedError

    def supervised_samples(
        self,
        queries: List[str],
        ids: Union[List[List[ChunkId]], List[List[str]], List[List[int]]],
        uses_db_id: bool,
    ) -> Union[SupervisedBatch, CustomIdSupervisedBatch]:
        if uses_db_id:
            return SupervisedBatch(
                query=queries,
                chunk_id=ids,
            )
        return CustomIdSupervisedBatch(query=queries, custom_id=ids)
