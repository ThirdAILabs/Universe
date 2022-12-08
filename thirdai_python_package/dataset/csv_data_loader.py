from io import BytesIO
from typing import List, Optional
from urllib.parse import urlparse

import awswrangler as wr
import pandas as pd
from thirdai._thirdai.dataset import DataLoader


class CSVDataLoader(DataLoader):
    """CSV data loader that can be used to load from a cloud
    storage instance such at s3 and GCS.

    Args:
        storage_path: The cloud storage instance type. Supported options
            include: "s3" and "gcs"
        batch_size: Batch size
        aws_credentials_file: Path to a file containing AWS access key id
            and AWS secret key
        gcs_credentials_file: Path to a file containing GCS credentials.
            This is always a credentials.json file.
    """

    def __init__(
        self,
        storage_path: str,
        batch_size: int = 10000,
        aws_credentials_file: str = None,
        gcs_credentials_file: str = None,
    ) -> None:

        if gcs_credentials_file:
            # Pandas requires the GCS file system in order
            # to authenticate a read request from a GCS bucket
            import gcsfs

        super().__init__(target_batch_size=batch_size)
        self._storage_path = storage_path
        self._target_batch_size = batch_size
        self._aws_credentials = aws_credentials_file
        self._gcs_credentials = gcs_credentials_file

        parsed_path = urlparse(self._storage_path, allow_fragments=False)
        self._cloud_instance_type = parsed_path.scheme
        self.restart()

    def _get_line_iterator(self):
        if self._cloud_instance_type == "s3":
            for row in wr.s3.read_csv(
                self._storage_path, chunksize=1, dtype=str, header=None
            ):
                row_as_string = ",".join(row.astype(str).values.flatten())
                yield row_as_string

        elif self._cloud_instance_type == "gcs":
            if self._gcs_credentials:
                for row in pd.read_csv(
                    self._storage_path,
                    storage_options={"token": self._gcs_credentials},
                    dtype=str,
                    chunksize=1,
                ):
                    yield ",".join(row.astype(str).values.flatten())
            else:
                for row in pd.read_csv(self._storage_path, dtype=str, chunksize=1):
                    yield ",".join(row.astype(str).values.flatten())

    def next_batch(self) -> Optional[List[str]]:
        lines = []
        while len(lines) < self._target_batch_size:
            next_line = self.next_line()
            if next_line == None:
                break
            lines.append(next_line)

        return lines if len(lines) else None

    def next_line(self) -> Optional[str]:
        return next(self._line_iterator, None)

    def restart(self) -> None:
        self._line_iterator = self._get_line_iterator()

    def resource_name(self) -> str:
        return self._storage_path
