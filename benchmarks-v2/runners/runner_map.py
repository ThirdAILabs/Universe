from .bolt_fc import BoltFullyConnectedRunner, BoltV2FullyConnectedRunner
from .dlrm import DLRMRunner
from .query_reformulation import QueryReformulationRunner
from .temporal import TemporalRunner
from .udt import UDTRunner

runner_map = {
    "bolt_fc": BoltFullyConnectedRunner,
    "bolt_v2_fc": BoltV2FullyConnectedRunner,
    "dlrm": DLRMRunner,
    "udt": UDTRunner,
    "query_reformulation": QueryReformulationRunner,
    "temporal": TemporalRunner,
}
