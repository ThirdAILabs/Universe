from typing import Iterable, List, Union

import pandas as pd

from ..core.supervised import SupervisedDataset
from ..core.types import ChunkId, CustomIdSupervisedBatch, SupervisedBatch


class InMemorySupervised(SupervisedDataset):
    def __init__(
        self,
        queries: List[str],
        ids: Union[List[List[ChunkId]], List[List[str]], List[List[int]]],
        uses_db_id: bool,
    ):
        self.queries = pd.Series(queries)
        self.ids = pd.Series(ids)
        self.uses_db_id = uses_db_id

    def samples(
        self,
    ) -> Union[Iterable[SupervisedBatch], Iterable[CustomIdSupervisedBatch]]:
        return [self.supervised_samples(self.queries, self.ids, self.uses_db_id)]
