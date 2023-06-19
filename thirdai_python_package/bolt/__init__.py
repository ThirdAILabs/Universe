import thirdai._thirdai.bolt
from thirdai._thirdai.bolt import *

from .udt_modifications import (
    add_neural_index_aliases,
    modify_graph_udt,
    modify_mach_udt,
    modify_udt,
)

modify_udt()
modify_graph_udt()
modify_mach_udt()
add_neural_index_aliases()

__all__ = []
__all__.extend(dir(thirdai._thirdai.bolt))
