"""The ThirdAI Python package"""
__all__ = [
    "bolt",
    "search",
    "dataset",
    "hashing",
    "set_thirdai_license_path",
    "set_global_num_threads",
]

# Include these so we can use them just by import the top level.
import thirdai.bolt
import thirdai.search
import thirdai.dataset
import thirdai.hashing

# Import the top level methods so they are available directly from thirdai
# If the import fails it means this build doesn't expose these methods, so we
# just pass
try:
    from thirdai._thirdai import set_thirdai_license_path
except ImportError:
    pass
try:
    from thirdai._thirdai import set_global_num_threads
except ImportError:
    pass

# Don't import this or include it in __all__ for now because it requires
# pytorch + transformers.
# import thirdai.embeddings
