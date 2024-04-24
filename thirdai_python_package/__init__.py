"""The ThirdAI Python package"""

__all__ = [
    "bolt",
    "search",
    "dataset",
    "data",
    "hashing",
    "distributed_bolt",
    "licensing",
    "demos",
    "gen",
    "telemetry",
    "set_global_num_threads",
    "logging",
]

# Include these so we can use them just by import the top level.
import thirdai.bolt as bolt
import thirdai.data as data
import thirdai.dataset as dataset
import thirdai.demos as demos
import thirdai.gen as gen
import thirdai.hashing as hashing
import thirdai.licensing as licensing
import thirdai.search as search
import thirdai.telemetry as telemetry

# Relay __version__ from C++
from thirdai._thirdai import __version__, logging, set_seed

try:
    from thirdai._thirdai import set_global_num_threads

    __all__.extend(["set_global_num_threads"])
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
