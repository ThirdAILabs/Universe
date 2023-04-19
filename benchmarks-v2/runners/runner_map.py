from .bolt_fc import BoltFullyConnectedRunner
from .dlrm import DLRMRunner
from .query_reformulation import QueryReformulationRunner
from .temporal import TemporalRunner
from .udt import UDTRunner

runner_map = {
    "bolt_fc": BoltFullyConnectedRunner,
    "dlrm": DLRMRunner,
    "udt": UDTRunner,
    "query_reformulation": QueryReformulationRunner,
    "temporal": TemporalRunner,
}
