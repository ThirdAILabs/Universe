import io

from thirdai._thirdai.dataset import DataLoader


class ParquetLoader(DataLoader):
    def __init__(self, parquet_path, batch_size):
        DataLoader.__init__(self, batch_size)

        import pyarrow.parquet as pq

        self._parquet_path = parquet_path
        self._parquet_table = pq.read_table(parquet_path)
        self._batch_size = batch_size
        self.restart()

    def restart(self):
        self._line_iterator = self._get_line_iterator()

    def _get_line_iterator(self):
        from pyarrow import csv

        first = True
        for single_line_batch in self._parquet_table.to_batches(1):
            buf = io.BytesIO()
            csv.write_csv(single_line_batch, buf)
            buf.seek(0)
            header, data_line, *_ = buf.read().decode().split("\n")
            if first:
                yield header + "\n"
                first = False
            yield data_line + "\n"

    def next_batch(self):
        lines = []
        while len(lines) < self._batch_size:
            next_line = self.next_line()
            if next_line == None:
                break
            lines.append(next_line)
        if lines == []:
            return None
        return lines

    def next_line(self):
        next_line = next(self._line_iterator, None)
        return next_line

    def resource_name(self):
        return self._parquet_path
