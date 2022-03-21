from .source_interfaces import SourceLocation

class LocalFileSystem(SourceLocation):
  def __init__(self, path_to_file: str):
    self.__path_to_file = path_to_file
    self.__file = None

  def open(self):
    self.__file = open(self.__path_to_file)
    return self.__file
  
  def close(self):
    if self.__file is not None:
      self.__file.close()
      self.__file = None