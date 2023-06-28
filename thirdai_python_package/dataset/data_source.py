from typing import List, Optional

from thirdai._thirdai.dataset import DataSource


class PyDataSource(DataSource):
    """Base class for DataSources Implemented in Python.
    Implements common methods `next_batch`, `next_line`, and `restart`.
    Concrete implementations must implement `_get_line_iterator` and
    `resource_name`.
    """

    def __init__(self):
        DataSource.__init__(self)

    def _get_line_iterator(self):
        raise NotImplementedError()

    def resource_name(self) -> str:
        raise NotImplementedError()

    def next_batch(self, target_batch_size) -> Optional[List[str]]:
        lines = []
        while len(lines) < target_batch_size:
            next_line = self.next_line()
            if next_line == None:
                break
            lines.append(next_line)

        return lines if len(lines) else None

    def next_line(self) -> Optional[str]:
        next_line = next(self._line_iterator, None)
        return next_line

    def restart(self) -> None:
        self._line_iterator = self._get_line_iterator()
