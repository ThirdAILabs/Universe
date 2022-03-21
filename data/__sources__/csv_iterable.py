from typing import Generator, List
from .source_interfaces import SourceFormat

class CsvIterable(SourceFormat):
  def __init__(self, delimiter=','):
    self.__delimiter = delimiter
    return
  
  def rows(self, file) -> Generator[List[str], None, None]:
    for line in file:
      yield line.split(self.__delimiter)

    # If exhausted
    while True:
      yield None