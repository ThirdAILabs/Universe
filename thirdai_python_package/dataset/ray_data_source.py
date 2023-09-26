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

    def __init__(self, ray_dataset, model_target_column, document_target_column):
        PyDataSource.__init__(self)
        self.ray_dataset = ray_dataset
        self.model_target_column = model_target_column
        self.document_target_column = document_target_column
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
        data_dict = {}
        for column_name in column_names:
            if column_name == self.document_target_column:
                data_dict[self.model_target_column] = [self.model_target_column]
            else:
                data_dict[column_name] = [column_name]

        yield pd.DataFrame(data_dict).to_csv(index=None, header=None)
        # return row-by-row data
        for chunk in self.ray_dataset.iter_batches(
            batch_size=self.DEFAULT_CHUNK_SIZE, batch_format="pandas"
        ):
            for i in range(len(chunk)):
                # TODO(pratik): Ray dataset lacks built-in column renaming support.
                # Reference: https://docs.ray.io/en/latest/data/api/dataset.html#basic-transformations
                # Only add/drop column options available, but they could be memory-intensive.
                yield (
                    chunk.iloc[i : i + 1]
                    .rename(
                        columns={self.document_target_column: self.model_target_column}
                    )
                    .to_csv(header=None, index=None)
                    .strip("\n")
                )

    def resource_name(self) -> str:
        return f"ray-dataset-sources"
