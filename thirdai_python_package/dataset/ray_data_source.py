import pandas as pd
from thirdai.dataset.data_source import PyDataSource


class RayCsvDataSource(PyDataSource):
    """
    RayCsvDataSource ingests ray datasets during distributed training.
    Using this ideally we should be able to load data from any of
    the sources mentioned here https://docs.ray.io/en/latest/data/loading-data.html
    which includes, parquet, s3, gcs, dask, spark, sql etc. It should work
    out of the box for single amchine training too.
    """

    DEFAULT_CHUNK_SIZE = 1000

    def __init__(self, ray_dataset):
        PyDataSource.__init__(self)
        self.ray_dataset = ray_dataset
        self.restart()
        try:
            import ray
        except ImportError:
            raise ImportError(
                "ray is not installed. Please install it to use RayCsvDataSource."
            )

    def _get_line_iterator(self):
        # return the header first
        column_names = self.ray_dataset.schema().names
        yield pd.DataFrame(
            {column_name: [column_name] for column_name in column_names}
        ).to_csv(index=None, header=None)
        # return row-by-row data
        for chunk in self.ray_dataset.iter_batches(
            batch_size=self.DEFAULT_CHUNK_SIZE, batch_format="pandas"
        ):
            for i in range(len(chunk)):
                yield chunk.iloc[i : i + 1].to_csv(header=None, index=None).strip("\n")

    def resource_name(self) -> str:
        return f"ray-dataset-sources"
