from .bolt_fc import BoltFullyConnectedRunner
from .distributed import DistributedRunner
from .distributed_ndb import DistributedNDBRunner
from .dlrm import DLRMRunner
from .mini_benchmark_runners.query_reformulation import (
    MiniBenchmarkQueryReformulationRunner,
)
from .mini_benchmark_runners.temporal import MiniBenchmarkTemporalRunner
from .mini_benchmark_runners.udt import MiniBenchmarkUDTRunner
from .ndb_runner import NDBRunner
from .query_reformulation import QueryReformulationRunner
from .rlhf import RlhfRunner
from .temporal import TemporalRunner
from .udt import UDTRunner

runner_map = {
    "bolt_fc": BoltFullyConnectedRunner,
    "dlrm": DLRMRunner,
    "udt": UDTRunner,
    "query_reformulation": QueryReformulationRunner,
    "temporal": TemporalRunner,
    "mini_benchmark_udt": MiniBenchmarkUDTRunner,
    "mini_benchmark_query_reformulation": MiniBenchmarkQueryReformulationRunner,
    "mini_benchmark_temporal": MiniBenchmarkTemporalRunner,
    "distributed": DistributedRunner,
    "rlhf": RlhfRunner,
    "distributed_ndb": DistributedNDBRunner,
    "neural_db": NDBRunner,
}
