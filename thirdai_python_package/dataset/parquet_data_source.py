import io

from thirdai.dataset.data_source import PyDataSource


class ParquetSource(PyDataSource):
    def __init__(self, parquet_path):
        PyDataSource.__init__(self)

        # By importing here, we make it so that pyarrow isn't a dependency.
        # If pyarrow isn't installed and you try to read a parquet, this will
        # throw an import error. This is standard practice, see e.g.
        # https://pandas.pydata.org/docs/getting_started/install.html
        import pyarrow.parquet as pq

        self._parquet_path = parquet_path
        self._parquet_table = pq.read_table(parquet_path)
        self.restart()

    def _get_line_iterator(self):
        from pyarrow import csv

        # This loop is pretty hacky. Basically, as a quick overview, we loop
        # through the parquet file line by line, write each line to an in memory
        # buffer as a two line csv (header and single line of data) using the
        # pyarrow csv writer, and then yield string versions of the data line
        # (and header line if it's the first line)
        first = True
        for single_line_batch in self._parquet_table.to_batches(1):
            buf = io.BytesIO()
            csv.write_csv(single_line_batch, buf)
            buf.seek(0)

            header, data_line, *_ = buf.read().decode().split("\n")

            if first:
                yield header
                first = False
            yield data_line

    def resource_name(self):
        return self._parquet_path
