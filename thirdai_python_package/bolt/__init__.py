import thirdai._thirdai.bolt
from thirdai._thirdai.bolt import *

from .ner_modifications import modify_ner
from .udt_modifications import (
    modify_graph_udt,
    modify_mach_udt,
    modify_udt,
    modify_udt_constructor,
)

modify_udt()
modify_graph_udt()
modify_mach_udt()
modify_ner()
modify_udt_constructor()


__all__ = []
__all__.extend(dir(thirdai._thirdai.bolt))
