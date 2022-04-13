from typing import Iterator, List


class Parser:
    """Interface for an object that defines how individual samples (rows) are streamed
    from the the data source and parsed into a row of features.
    """

    # TODO(Geordie): Support returning lists of other types.
    def rows(self, source: any) -> Iterator[List[str]]:
        """Streams the next sample from the data source and parses it into a row of
        features.

        Arguments:
          source: depends on the concrete implementation - the data source
            returned by Source.open()
        """
        yield
