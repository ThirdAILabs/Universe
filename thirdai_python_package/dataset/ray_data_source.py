import os
import ray

import pandas as pd
from thirdai.dataset.data_source import PyDataSource


class RayDataSource(PyDataSource):
    def __init__(self, ray_dataset):
        PyDataSource.__init__(self)
        self.ray_dataset = ray_dataset

    def _get_line_iterator(self):
        yield pd.DataFrame(self.ray_dataset.columns()).to_csv(index=False, header=True)

        for row in self.ray_dataset.iter_rows():
            yield row.to_pandas().to_csv(header=None, index=None).strip("\n")

    def resource_name(self) -> str:
        return f"Ray-Dataset-sources-{self.ray_dataset.input_files()}-count-{self.ray_dataset.count()}"
