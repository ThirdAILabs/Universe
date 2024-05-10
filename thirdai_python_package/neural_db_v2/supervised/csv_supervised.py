from typing import Iterable, Optional, Union

import pandas as pd

from ..core.supervised import SupervisedDataset
from ..core.types import CustomIdSupervisedBatch, SupervisedBatch


class CsvSupervised(SupervisedDataset):
    def __init__(
        self,
        path: str,
        query_column: str,
        id_column: str,
        id_delimiter: str,
        uses_db_id: bool,
        custom_ids_as_strings: bool = True,
    ):
        self.path = path
        self.query_column = query_column
        self.id_column = id_column
        self.id_delimiter = id_delimiter
        self.uses_db_id = uses_db_id
        self.custom_ids_as_strings = custom_ids_as_strings

    def samples(
        self,
    ) -> Union[Iterable[SupervisedBatch], Iterable[CustomIdSupervisedBatch]]:
        df = pd.read_csv(self.path)

        ids = df[self.id_column].map(lambda val: str(val).split(self.id_delimiter))

        if self.uses_db_id:
            ids = pd.Series([list(map(int, row_ids)) for row_ids in ids])
        elif not self.custom_ids_as_strings:
            # We will attempt to cast custom ids to integers unless the user
            # explicitly specifies to keep them as strings.
            try:
                ids = pd.Series([list(map(int, row_ids)) for row_ids in ids])
            except:
                pass

        return [
            self.supervised_samples(
                queries=df[self.query_column],
                ids=ids,
                uses_db_id=self.uses_db_id,
            )
        ]
