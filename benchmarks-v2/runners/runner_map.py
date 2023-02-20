from .bolt_fc import BoltFullyConnectedRunner
from .dlrm import DLRMRunner
from .udt import UDTRunner

runner_map = {
    "bolt_fc": BoltFullyConnectedRunner,
    "dlrm": DLRMRunner,
    "udt": UDTRunner,
}
