import json

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


class RayTextDataSource(PyDataSource):
    def __init__(self, ray_dataset, tokenize_for_pretraining=False):
        PyDataSource.__init__(self)
        self.ray_dataset = ray_dataset
        self.tokenize_for_pretraining = tokenize_for_pretraining
        try:
            import ray
            from transformers import GPT2Tokenizer
        except ImportError:
            raise ImportError(
                "This class requires both the 'ray' and 'transformers' libraries. Please ensure they are installed."
            )
        if self.tokenize_for_pretraining:
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.restart()

    def _get_line_iterator(self):
        for row in self.ray_dataset.iter_rows():
            text = row["text"]
            if self.tokenize_for_pretraining:
                text = self._tokenize(text)
            yield text

    def _tokenize(self, text):
        tokens = self.tokenizer.encode(text)
        tokenized_text = " ".join(map(str, tokens))
        json_output = json.dumps({"target": tokenized_text})
        return json_output

    def resource_name(self) -> str:
        return f"ray-dataset-sources"
