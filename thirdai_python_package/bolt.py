import thirdai._thirdai.bolt
from thirdai._thirdai.bolt import *
__all__ = []
__all__.extend(dir(thirdai._thirdai.bolt))

try:
  # This is a hack to only define datastructures for internal builds. 
  from thirdai._thirdai.bolt import nn
  import thirdai._bolt_datastructures
  from thirdai._bolt_datastructures import *
  __all__.extend(dir(thirdai._bolt_datastructures))
except:
  pass
