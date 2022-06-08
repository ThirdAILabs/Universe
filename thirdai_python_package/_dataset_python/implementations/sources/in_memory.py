from ...interfaces import Source


class InMemoryCollection(Source):
    """Used when the dataset is already loaded into memory
    and assigned to a variable, e.g. a list or a numpy array.
    """

    def __init__(self, obj: any) -> None:
        """Constructor.

        Arguments:
          obj: any object in memory.
        """
        self.obj = obj

    def open(self):
        """Returns the accepted object."""
        return self.obj

    def close(self):
        """Does nothing."""
        return
