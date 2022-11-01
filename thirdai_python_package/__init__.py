"""The ThirdAI Python package"""
__all__ = [
    "bolt",
    "search",
    "dataset",
    "data",
    "hashing",
    "distributed_bolt",
    "set_thirdai_license_path",
    "set_global_num_threads",
    "logging",
]

# Include these so we can use them just by import the top level.
import thirdai.bolt as bolt
import thirdai.data as data
import thirdai.dataset as dataset
import thirdai.hashing as hashing
import thirdai.search as search

# Relay __version__ from C++
from thirdai._thirdai import __version__, logging

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

# ray's grcpio dependency installation is not trivial on
# Apple Mac M1 Silicon and requires conda.
#
# See:
# [1] https://github.com/grpc/grpc/issues/25082,
# [2] https://docs.ray.io/en/latest/ray-overview/installation.html#m1-mac-apple-silicon-support
# For the time being users are expected to explictly import the package.
#
# TODO(pratkpranav): Uncomment the following when this issue is solved upstream.
# import thirdai.distributed_bolt


# Don't import this or include it in __all__ for now because it requires
# pytorch + transformers.
# import thirdai.embeddings
