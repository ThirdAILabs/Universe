from ...interfaces import Source


class LocalFileSystem(Source):
    """Used for loading a dataset from the local file system."""

    def __init__(self, path_to_file: str):
        """Constructor:

        Arguments:
          path_to_file: string - the path to the file in the
            local file system.
        """
        self.__path_to_file = path_to_file
        self.__file = None

    def open(self):
        """Opens the file."""
        self.__file = open(self.__path_to_file)
        return self.__file

    def close(self):
        """Closes the file."""
        if self.__file is not None:
            self.__file.close()
            self.__file = None
