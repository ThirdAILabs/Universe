from .bolt_fc import BoltFullyConnectedRunner, BoltV2FullyConnectedRunner
from .distributed_v1 import DistributedRunner_v1
from .distributed_v2 import DistributedRunner_v2
from .dlrm import DLRMRunner, DLRMV2Runner
from .mini_benchmark_runners.query_reformulation import (
    MiniBenchmarkQueryReformulationRunner,
)
from .mini_benchmark_runners.temporal import MiniBenchmarkTemporalRunner
from .mini_benchmark_runners.udt import MiniBenchmarkUDTRunner
from .query_reformulation import QueryReformulationRunner
from .temporal import TemporalRunner
from .udt import UDTRunner

runner_map = {
    "bolt_fc": BoltFullyConnectedRunner,
    "bolt_v2_fc": BoltV2FullyConnectedRunner,
    "dlrm": DLRMRunner,
    "dlrm_v2": DLRMV2Runner,
    "udt": UDTRunner,
    "query_reformulation": QueryReformulationRunner,
    "temporal": TemporalRunner,
    "mini_benchmark_udt": MiniBenchmarkUDTRunner,
    "mini_benchmark_query_reformulation": MiniBenchmarkQueryReformulationRunner,
    "mini_benchmark_temporal": MiniBenchmarkTemporalRunner,
    "distributed_v1": DistributedRunner_v1,
    "distributed_v2": DistributedRunner_v2,
}
