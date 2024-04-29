from abc import ABC, abstractmethod
from typing import Iterable, List, Union

from .types import ChunkId, CustomIdSupervisedBatch, SupervisedBatch


class Supervised(ABC):
    @abstractmethod
    def samples(
        self,
    ) -> Union[Iterable[SupervisedBatch], Iterable[CustomIdSupervisedBatch]]:
        pass

    def supervised_samples(
        queries: List[str],
        ids: Union[List[List[ChunkId]], List[List[str]], List[List[int]]],
        uses_db_id: bool,
    ) -> Union[SupervisedBatch, CustomIdSupervisedBatch]:
        if uses_db_id:
            return SupervisedBatch(query=queries, chunk_id=ids)
        return CustomIdSupervisedBatch(query=queries, custom_id=ids)
