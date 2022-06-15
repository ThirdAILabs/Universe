from typing import Iterator, List, Iterable, Optional
from ...interfaces import Parser


class CsvIterable(Parser):
    """Streams samples (rows) from an iterable of CSV strings."""

    def __init__(self, delimiter=","):
        """Constructor.

        Arguments:
          delimiter: str - CSV delimiter.
        """
        self.__delimiter = delimiter
        return

    def rows(self, source: Iterable[str]) -> Iterator[Optional[List[str]]]:
        """Streams the next sample from a CSV iterable and parses it into
        a row of features.

        Arguments:
          source: iterable of strings - the data source.
        """
        for line in source:
            # Each line is split at every occurrence of the
            # specified delimiter, yielding a list of strings
            # for each line.
            yield line.split(self.__delimiter)
