from typing import Iterable, List, Union

import pandas as pd

from ..core.supervised import SupervisedDataset
from ..core.types import ChunkId, SupervisedBatch


class InMemorySupervised(SupervisedDataset):
    def __init__(
        self,
        queries: List[str],
        ids: Union[List[List[ChunkId]], List[List[str]], List[List[int]]],
    ):
        self.queries = pd.Series(queries)
        self.ids = pd.Series(ids)

    def samples(self) -> Iterable[SupervisedBatch]:
        return [self.supervised_samples(self.queries, self.ids)]
