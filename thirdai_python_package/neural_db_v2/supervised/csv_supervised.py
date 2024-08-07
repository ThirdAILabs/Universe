from typing import Iterable

import pandas as pd

from ..core.supervised import SupervisedDataset
from ..core.types import SupervisedBatch


class CsvSupervised(SupervisedDataset):
    def __init__(
        self,
        path: str,
        query_column: str,
        id_column: str,
        id_delimiter: str,
    ):
        self.path = path
        self.query_column = query_column
        self.id_column = id_column
        self.id_delimiter = id_delimiter

    def samples(self) -> Iterable[SupervisedBatch]:
        df = pd.read_csv(self.path)

        ids = df[self.id_column].map(
            lambda val: list(map(int, str(val).split(self.id_delimiter)))
        )

        return [self.supervised_samples(queries=df[self.query_column], ids=ids)]
