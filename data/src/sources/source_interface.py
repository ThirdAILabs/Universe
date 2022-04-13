class Source:
    """Interface for an object that defines how the dataset is accessed, e.g.
    through a database connector or through the local file system.
    """

    def open(self):
        """Opens the data source."""
        return

    def close(self):
        """Closes the data source."""
        return
