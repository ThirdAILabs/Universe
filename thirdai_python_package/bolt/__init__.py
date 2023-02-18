import thirdai._thirdai.bolt
from thirdai._thirdai.bolt import *

from .udt_modifications import modify_udt_classifier, modify_udt

modify_udt_classifier()
modify_udt()

__all__ = []
__all__.extend(dir(thirdai._thirdai.bolt))
