from typing import Iterator, List, Iterable, Optional
from parser_interface import Parser


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
            yield line.split(self.__delimiter)

        # If exhausted
        while True:
            yield None
