from typing import Generator, List

class SourceLocation:
  """Abstract class for data source location. E.g. S3, local file system, python array
  """
  def open(self):
    """Opens the data file.
    """
    return
  
  def close(self):
    """Closes the data file.
    """
    return

class SourceFormat:
  """Abstract class for data source format. E.g. CSV, or Parquet
  """
  def rows(self, file) -> Generator[List[str],None,None]:
    """Yields the columns of the next row.
    """
    yield