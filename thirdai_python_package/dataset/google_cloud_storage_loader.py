from typing import List, Optional
from thirdai._thirdai.dataset import DataLoader
import pandas as pd
from google.cloud import storage 

class GCSDataLoader(DataLoader):
    """
    This class handles loading data from Google Cloud Storage (GCS) bucket.
    """

    def __init__(self, bucket_name: str, batch_size: str = 10000) -> None:
        super().__init__(target_batch_size=batch_size)
        self._bucket_name = bucket_name
        self._batch_size = batch_size

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

    def next_line(self) -> Optional[str]:
        pass

    def resource_name(self) -> str:
        return self._gcs_bucket_path

    def restart(self) -> None:
        pass


if __name__=="__main__":
    file_path = "gcs://thirdai-udt-service-bucket/data/inference_batch.csv"
    df = pd.read_csv(file_path, storage_options={"token": "~/Desktop/credentials.json"})

    print(df)