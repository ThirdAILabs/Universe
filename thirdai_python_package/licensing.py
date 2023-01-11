try:
    import thirdai._thirdai.licensing
    from thirdai._thirdai.licensing import *

    __all__ = []
    __all__.extend(dir(thirdai._thirdai.licensing))
except ImportError:
    pass
