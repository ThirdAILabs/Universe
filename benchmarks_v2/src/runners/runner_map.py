from .bolt_fc import BoltFullyConnectedRunner
from .dlrm import DLRMRunner
from .query_reformulation import QueryReformulationRunner
from .temporal import TemporalRunner
from .udt import UDTRunner
from .mini_benchmark_runners.udt import MiniBenchmarkUDTRunner
from .mini_benchmark_runners.query_reformulation import MiniBenchmarkQueryReformulationRunner
from .mini_benchmark_runners.temporal import MiniBenchmarkTemporalRunner

runner_map = {
    "bolt_fc": BoltFullyConnectedRunner,
    "dlrm": DLRMRunner,
    "udt": UDTRunner,
    "query_reformulation": QueryReformulationRunner,
    "temporal": TemporalRunner,
    "mini_benchmark_udt": MiniBenchmarkUDTRunner,
    "mini_benchmark_query_reformulation": MiniBenchmarkQueryReformulationRunner,
    "mini_benchmark_temporal": MiniBenchmarkTemporalRunner
}