from typing import Iterable, Union

import pandas as pd

from ..core.supervised import Supervised
from ..core.types import CustomIdSupervisedBatch, SupervisedBatch


class CsvSupervised(Supervised):
    def __init__(
        self,
        path: str,
        query_column: str,
        id_column: str,
        id_delimiter: str,
        uses_db_id: bool,
        max_rows: int = 10_000_000,
    ):
        self.path = path
        self.query_column = query_column
        self.id_column = id_column
        self.id_delimiter = id_delimiter
        self.uses_db_id = uses_db_id
        self.max_rows = max_rows

    def samples(
        self,
    ) -> Union[Iterable[SupervisedBatch], Iterable[CustomIdSupervisedBatch]]:
        data_iter = pd.read_csv(self.path, chunksize=self.max_rows)

        for df in data_iter:
            df.reset_index(drop=True, inplace=True)

            # TODO(david) do we need some sort of assertion for having list of values?
            # or do we need to convert to list of values somewhere?
            # I don't think we need to because the data pipeline should take care of it but we might need to update the typing

            yield self.supervised_samples(
                queries=df[self.query_column],
                ids=df[self.id_column],
                uses_db_id=self.uses_db_id,
            )
