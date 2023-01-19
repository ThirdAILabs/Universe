from io import BytesIO
from typing import List, Optional
from urllib.parse import urlparse

import pandas as pd
from thirdai._thirdai.dataset import DataSource


class CSVDataSource(DataSource):
    """CSV data source that can be used to load from a cloud
    storage instance such as s3 and GCS.

    Args:
        storage_path: Path to the CSV file.
        batch_size: Batch size
        gcs_credentials_path: Path to a file containing GCS credentials.
            This is typically a credentials.json file. For the authorization
            protocol to work, the credentials file must contain a project ID,
            client E-mail, a token URI and a private key.

    Note: To read a file from s3, Pandas will expect a credentials file
        containing an AWS access key id and an AWS secret key.
    """

    DEFAULT_CHUNK_SIZE = 1000

    def __init__(
        self,
        storage_path: str,
        batch_size: int = 10000,
        gcs_credentials_path: str = None,
    ) -> None:

        if gcs_credentials_path:
            # Pandas requires the GCS file system in order
            # to authenticate a read request from a GCS bucket
            import gcsfs

        super().__init__(target_batch_size=batch_size)
        self._storage_path = storage_path
        self._target_batch_size = batch_size
        self._gcs_credentials = gcs_credentials_path
        self._storage_options = (
            {"token": gcs_credentials_path} if gcs_credentials_path else None
        )

        parsed_path = urlparse(self._storage_path, allow_fragments=False)
        self._cloud_instance_type = parsed_path.scheme
        self.restart()

    def _get_line_iterator(self):
        if self._cloud_instance_type not in ["s3", "gcs"]:
            raise ValueError(
                f"Invalid data storage path starting with {self._storage_path}"
            )

        for chunk in pd.read_csv(
            self._storage_path,
            chunksize=self.DEFAULT_CHUNK_SIZE,
            storage_options=self._storage_options,
            dtype="object",
            header=None,
        ):
            for _, row in chunk.iterrows():
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
