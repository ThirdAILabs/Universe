from typing import Iterator, List


class SourceLocation:
    """Interface for an object that defines how the dataset is accessed, e.g.
    through a database connector or through the local file system.
    """

    def open(self):
        """Opens the data source."""
        return

    def close(self):
        """Closes the data source."""
        return


class SourceFormat:
    """Interface for an object that defines how individual samples (rows) are streamed
    from the the data source and parsed into a row of features.
    """

    def rows(self, source: any) -> Iterator[List[str]]:
        """Streams the next sample from the data source and parses it into a row of
        features.

        Arguments:
          source: depends on the concrete implementation - the data source
            returned by SourceLocation.open()
        """
        yield
