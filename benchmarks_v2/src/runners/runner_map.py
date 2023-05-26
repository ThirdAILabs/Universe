from .backward_compatibility_runners.query_reformulation import (
    BackwardCompatibilityQueryReformulationRunner,
)
from .backward_compatibility_runners.temporal import BackwardCompatibilityTemporalRunner
from .backward_compatibility_runners.udt import BackwardCompatibilityUDTRunner
from .bolt_fc import BoltFullyConnectedRunner, BoltV2FullyConnectedRunner
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
    "backward_compatibility_udt": BackwardCompatibilityUDTRunner,
    "backward_compatibility_query_reformulation": BackwardCompatibilityQueryReformulationRunner,
    "backward_compatibility_temporal": BackwardCompatibilityTemporalRunner,
}
