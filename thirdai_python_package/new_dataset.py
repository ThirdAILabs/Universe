import thirdai._thirdai.new_dataset
from thirdai._thirdai.new_dataset import *

from abc import ABC, abstractmethod
from typing import Optional

class ColumnMapGenerator(ABC):
    @abstractmethod
    def next() -> Optional[ColumnMap]:
        pass

    @abstractmethod
    def restart() -> None:
        pass



__all__ = ["ColumnMapGenerator"]
__all__.extend(dir(thirdai._thirdai.new_dataset))
