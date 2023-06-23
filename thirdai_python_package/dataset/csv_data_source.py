import os
from io import BytesIO
from typing import List, Optional
from urllib.parse import urlparse

import pandas as pd
from thirdai.dataset.data_source import PyDataSource


class CSVDataSource(PyDataSource):
    """CSV data source that can be used to load from a cloud
    storage instance such as s3 and GCS.

    Args:
        storage_path: Path to the CSV file.
        batch_size: Batch size
        gcs_credentials_path: Path to a file containing GCS credentials.
            This is typically a credentials file. For the authorization
            protocol to work, the credentials file must contain a project ID,
            client E-mail, a token URI and a private key.

    Note: To read a file from s3, Pandas will expect a credentials file
        containing an AWS access key id and an AWS secret key located at
        ~/aws/credentials. For GCS, the gcloud CLI typically stores the
        credentials file in locations, such as ~/.config/gcloud/credentials
        or ~/.config/gcloud/application_default_credentials.json.
        https://gcsfs.readthedocs.io/en/latest/api.html
    """

    DEFAULT_CHUNK_SIZE = 1000

    # These are provided here since pandas.read_csv does not implicitly use these
    # paths although google cloud recommends storing credentials in either one of them.
    # It is only for s3 that pandas.read_csv implicitly searches for a ~/.aws/credentials file.
    FIRST_DEFAULT_GCS_CREDS_PATH = (
        "~/.config/gcloud/application_default_credentials.json"
    )
    SECOND_DEFAULT_GCS_CREDS_PATH = "~/.config/gcloud/credentials"

    def __init__(
        self,
        storage_path: str,
        gcs_credentials_path: str = None,
    ) -> None:
        PyDataSource.__init__(self)

        if gcs_credentials_path:
            # Pandas requires the GCS file system in order
            # to authenticate a read request from a GCS bucket
            import gcsfs

        self._storage_path = storage_path
        self._gcs_credentials = gcs_credentials_path

        token = None
        if gcs_credentials_path:
            token = gcs_credentials_path
        else:
            if os.path.exists(self.FIRST_DEFAULT_GCS_CREDS_PATH):
                token = self.FIRST_DEFAULT_GCS_CREDS_PATH
            elif os.path.exists(self.SECOND_DEFAULT_GCS_CREDS_PATH):
                token = self.SECOND_DEFAULT_GCS_CREDS_PATH

        self._storage_options = {"token": token}

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
            for i in range(len(chunk)):
                yield chunk.iloc[i : i + 1].to_csv(header=None, index=None).strip("\n")

    def resource_name(self) -> str:
        return self._storage_path
