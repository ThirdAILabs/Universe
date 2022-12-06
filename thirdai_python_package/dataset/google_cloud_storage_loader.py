import socket
from io import BytesIO
from typing import List, Optional

import pandas as pd
from thirdai._thirdai.dataset import DataLoader


class GCSDataLoader(DataLoader):
    """Data Loader class for Google Cloud Storage

    The google cloud storage client and the google cloud library
    that verifies credentials are imported here so that they are
    not dependencies of our package. These are required in order
    to read from a GCS bucket with pandas.

    In case this class is used in a local environment, the user
    has to provide credentials that are verified via a OAuth 2.0 
    access token. 

    Args:
        bucket_name: The name of the bucket in the Google Cloud Storage (GCS)
            instance
        resource_path: This is the path to the file to be read from the bucket
        batch_size: batch size 
        gcp_credentials: Credentials required to authorize any CRUD operations 
            on a GCS instance. In a local environment, this is supposed to be 
            a credentials.json file that a user can download from their dashboard
            once they create a project on Google Cloud. In a cloud environment, 
            the credentials are not needed.
        file_format: Extension of the file to be read. Options include `csv`, 
            `parquet` and `pqt`. 

    """

    def __init__(
        self,
        bucket_name: str,
        resource_path: str,
        batch_size: str = 10000,
        gcp_credentials: Optional[str] = None,
        file_format: str = "csv",
    ) -> None:

        from google.cloud import storage
        from google.oauth2 import service_account

        super().__init__(target_batch_size=batch_size)
        self._resource_path = resource_path
        self._batch_size = batch_size
        self.file_format = file_format

        if self._is_google_cloud_instance():
            self._storage_client = storage.Client()
        else:
            if not gcp_credentials:
                raise ValueError(
                    "credentials.json File Required for Authentication to the "
                    "Google Cloud Storage."
                )
            credentials = service_account.Credentials.from_service_account_file(
                filename=gcp_credentials
            )
            self._storage_client = storage.Client(credentials=credentials)

        bucket = self._storage_client.get_bucket(bucket_name)
        blob = bucket.get_blob(blob_name=resource_path)
        self._binary_stream = blob.download_as_string()

        self.restart()

    def next_batch(self) -> Optional[List[str]]:
        lines = []
        while len(lines) < self._target_batch_size:
            next_line = self.next_line()
            if next_line == None:
                break
            lines.append(next_line)

        if len(lines) == 0:
            return None
        return lines

    def _get_line_iterator(self):
        if self.file_format == "csv":
            for row in pd.read_csv(BytesIO(self._binary_stream), chunksize=1):
                row_as_string = ",".join(row.astype(str).values.flatten())
                yield row_as_string

        elif self.file_format == "parquet" or self.file_format == "pqt":
            for row in pd.read_parquet(BytesIO(self._binary_stream, chunk_size=1)):
                row_as_string = ",".join(row.astype(str).values.flatten())
                yield row_as_string

    @staticmethod
    def _is_google_cloud_instance() -> bool:
        """
        Checks if the underlying environment is a google compute engine (GCE)
        instance via a DNS lookup to metadata server. This is needed to 
        ensure that in a local environment the path to a credentials file 
        is provided. 
        """
        try:
            socket.getaddrinfo("metadata.google.internal", 80)
        except socket.gaierror:
            return False
        return True

    def next_line(self) -> Optional[str]:
        return next(self._line_iterator, None)

    def resource_name(self) -> str:
        return f"gcs://{self._bucket_name}/{self._resource_path}"

    def restart(self) -> None:
        self._line_iterator = self._get_line_iterator()
