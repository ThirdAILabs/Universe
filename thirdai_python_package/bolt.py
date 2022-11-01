import thirdai._thirdai.bolt
from thirdai._thirdai.bolt import *

__all__ = []
__all__.extend(dir(thirdai._thirdai.bolt))

try:
    # This is a hack to only define datastructures for internal builds. For release builds
    # thirdai._thirdai.bolt.nn is undefined so we catch that exception and skip adding
    # the datastructures to the bolt submodule.
    import thirdai._bolt_datastructures
    from thirdai._bolt_datastructures import *
    from thirdai._thirdai.bolt import nn

    __all__.extend(dir(thirdai._bolt_datastructures))
except ImportError:
    pass
