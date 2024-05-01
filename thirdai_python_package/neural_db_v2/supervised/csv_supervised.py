from typing import Iterable, Optional, Union

import pandas as pd

from ..core.supervised import Supervised
from ..core.types import CustomIdSupervisedBatch, SupervisedBatch


class CsvSupervised(Supervised):
    def __init__(
        self,
        path: str,
        query_column: str,
        id_column: str,
        uses_db_id: bool,
        id_delimiter: Optional[str] = None,
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
        df = pd.read_csv(self.path)

        return [
            self.supervised_samples(
                queries=df[self.query_column],
                ids=df[self.id_column].map(
                    lambda val: list(str(val).split(self.id_delimiter))
                ),
                uses_db_id=self.uses_db_id,
            )
        ]
