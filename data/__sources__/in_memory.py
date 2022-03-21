from typing import Generator, List
from .source_interfaces import SourceLocation, SourceFormat

class InMemoryCollection(SourceLocation):
  def __init__(self, obj) -> None:
    self.obj = obj
  
  def open(self):
    return self.obj
